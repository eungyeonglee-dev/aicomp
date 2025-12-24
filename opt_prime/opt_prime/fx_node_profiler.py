import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import fx


@dataclass
class _CudaEventPair:
    start: torch.cuda.Event
    end: torch.cuda.Event


class FxNodeTimeProfiler:
    """
    Profile per-FX-node compute time by interpreting a GraphModule node-by-node.

    This is meant for case (B): "count ops as FX nodes" (e.g., ~130 nodes).
    It measures ONLY what happens inside GraphModule execution, so pipeline send/recv is excluded
    as long as send/recv is not called inside the GraphModule itself.

    Grouping:
      - call_module: group by regex on node.target (string)
      - others(call_function/call_method): group is propagated from input nodes when unambiguous
    """

    def __init__(
        self,
        *,
        enabled: bool,
        device_index: Optional[int],
        group_name_regex: str = r"model_layers_(\d+)",
    ):
        self.enabled = enabled
        self.device_index = device_index
        self._group_re = re.compile(group_name_regex)

        # node -> group key (e.g., "12") for propagation
        self._node_group: Dict[fx.Node, Optional[str]] = {}

        # pending (recorded) timings for one micro-batch
        self._pending_cuda: List[Tuple[str, Optional[str], _CudaEventPair]] = []
        self._pending_cpu: List[Tuple[str, Optional[str], int, int]] = []  # (node_key, group, start_ns, end_ns)

        # aggregated per-step
        self.node_total_ms: Dict[str, float] = {}
        self.node_count: Dict[str, int] = {}
        self.group_total_ms: Dict[str, float] = {}
        self.group_count: Dict[str, int] = {}

        self._graph_node_count: int = 0
        self._timed_node_count: int = 0

    def _is_cuda(self) -> bool:
        return torch.cuda.is_available() and self.device_index is not None

    def reset_stats(self) -> None:
        self.node_total_ms.clear()
        self.node_count.clear()
        self.group_total_ms.clear()
        self.group_count.clear()
        self._graph_node_count = 0
        self._timed_node_count = 0

    def _extract_group_from_target(self, target: Any) -> Optional[str]:
        if not isinstance(target, str):
            return None
        # Common transformer-wide ops outside per-layer blocks
        # (HuggingFace Llama): embed tokens and lm_head
        if "embed_tokens" in target:
            return "embed"
        if "lm_head" in target:
            return "lm_head"
        m = self._group_re.search(target)
        if not m:
            return None
        return m.group(1) if m.lastindex else m.group(0)

    def _propagate_group(self, node: fx.Node, args, kwargs) -> Optional[str]:
        # Prefer call_module target parsing
        if node.op == "call_module":
            return self._extract_group_from_target(node.target)

        # Propagate from input nodes when unambiguous
        groups = set()
        for in_node in node.all_input_nodes:
            g = self._node_group.get(in_node)
            if g is not None:
                groups.add(g)
        if len(groups) == 1:
            return next(iter(groups))
        return None

    def _should_time(self, node: fx.Node) -> bool:
        # Exclude bookkeeping nodes
        if node.op in ("placeholder", "get_attr", "output"):
            return False
        return True

    def _node_key(self, node: fx.Node) -> str:
        # Stable-ish readable key
        if node.op == "call_module":
            return f"{node.op}:{node.target}"
        if node.op == "call_function":
            return f"{node.op}:{getattr(node.target, '__name__', str(node.target))}"
        if node.op == "call_method":
            return f"{node.op}:{node.target}"
        return f"{node.op}:{node.name}"

    def run(self, gm: fx.GraphModule, *args, **kwargs):
        """
        Execute GraphModule via FX Interpreter, recording node times.
        """
        if not self.enabled:
            return gm(*args, **kwargs)

        self._node_group = {}
        self._pending_cuda.clear()
        self._pending_cpu.clear()

        class _Interpreter(fx.Interpreter):
            def __init__(self, outer: "FxNodeTimeProfiler", module: fx.GraphModule):
                super().__init__(module)
                self._outer = outer

            def run_node(self, n: fx.Node) -> Any:
                # Propagate group label before running the node (inputs already computed)
                gk = self._outer._propagate_group(n, n.args, n.kwargs)
                self._outer._node_group[n] = gk

                if not self._outer._should_time(n):
                    return super().run_node(n)

                key = self._outer._node_key(n)

                if self._outer._is_cuda():
                    start_ev = torch.cuda.Event(enable_timing=True)
                    end_ev = torch.cuda.Event(enable_timing=True)
                    start_ev.record()
                    out = super().run_node(n)
                    end_ev.record()
                    self._outer._pending_cuda.append((key, gk, _CudaEventPair(start=start_ev, end=end_ev)))
                    self._outer._timed_node_count += 1
                    return out

                # CPU timing
                import time

                t0 = time.perf_counter_ns()
                out = super().run_node(n)
                t1 = time.perf_counter_ns()
                self._outer._pending_cpu.append((key, gk, t0, t1))
                self._outer._timed_node_count += 1
                return out

        itp = _Interpreter(self, gm)
        # Count all graph nodes once per run (for debugging "130 ops" expectation)
        self._graph_node_count = sum(1 for _ in gm.graph.nodes)
        return itp.run(*args, **kwargs)

    def flush(self) -> None:
        """
        Convert pending node timings into aggregated stats.
        Call once per micro-batch to avoid per-node synchronization.
        """
        if not self.enabled:
            return

        if self._is_cuda():
            torch.cuda.synchronize(self.device_index)
            for key, gk, pair in self._pending_cuda:
                ms = pair.start.elapsed_time(pair.end)
                self.node_total_ms[key] = self.node_total_ms.get(key, 0.0) + float(ms)
                self.node_count[key] = self.node_count.get(key, 0) + 1
                if gk is not None:
                    self.group_total_ms[gk] = self.group_total_ms.get(gk, 0.0) + float(ms)
                    self.group_count[gk] = self.group_count.get(gk, 0) + 1
            self._pending_cuda.clear()
            return

        for key, gk, t0, t1 in self._pending_cpu:
            ms = (t1 - t0) / 1_000_000.0
            self.node_total_ms[key] = self.node_total_ms.get(key, 0.0) + float(ms)
            self.node_count[key] = self.node_count.get(key, 0) + 1
            if gk is not None:
                self.group_total_ms[gk] = self.group_total_ms.get(gk, 0.0) + float(ms)
                self.group_count[gk] = self.group_count.get(gk, 0) + 1
        self._pending_cpu.clear()

    def report_group_lines(self) -> List[str]:
        if not self.group_total_ms:
            return []

        def _group_sort_key(k: str):
            if k == "embed":
                return (0, -1)
            if k == "lm_head":
                return (2, 10**9)
            m = re.findall(r"(\d+)", k)
            if m:
                return (1, int(m[-1]))
            return (3, 10**9)

        lines: List[str] = []
        for k in sorted(self.group_total_ms.keys(), key=_group_sort_key):
            total = self.group_total_ms[k]
            cnt = self.group_count.get(k, 0)
            lines.append(f"layer[{k}]: total={total:.3f}ms, fx_nodes={cnt}")
        return lines

    def report_debug_header(self) -> str:
        return f"fx_graph_nodes={self._graph_node_count}, timed_nodes={self._timed_node_count}"


