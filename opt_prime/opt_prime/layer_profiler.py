import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch


@dataclass
class _CudaEventPair:
    start: torch.cuda.Event
    end: torch.cuda.Event


class LayerTimeProfiler:
    """
    Measure per-layer (module) forward compute time using hooks.

    - Intended to exclude pipeline send/recv time by design: hooks run only during module forward.
    - CUDA timing uses events on the current stream.
    - We synchronize once per micro-batch flush to avoid per-layer sync overhead.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        device_index: Optional[int],
        match_name_regex: Optional[str] = r"^model_layers_\d+$",
        match_module_class_names: Optional[Iterable[str]] = None,
        group_name_regex: Optional[str] = None,
    ):
        self.enabled = enabled
        self.device_index = device_index
        self._name_re = re.compile(match_name_regex) if match_name_regex else None
        self._class_name_allow: Optional[Set[str]] = (
            set(match_module_class_names) if match_module_class_names is not None else None
        )
        self._group_re = re.compile(group_name_regex) if group_name_regex else None

        self._current_phase: Optional[str] = None
        self._current_mb: Optional[int] = None

        self._handles = []

        # inflight stacks: key -> list[start_marker]
        self._inflight_cuda: Dict[Tuple[str, str, int], List[torch.cuda.Event]] = {}
        self._inflight_cpu: Dict[Tuple[str, str, int], List[int]] = {}

        # pending completed measurements for current context
        self._pending_cuda: List[Tuple[str, _CudaEventPair]] = []
        self._pending_cpu: List[Tuple[str, int, int]] = []  # (name, start_ns, end_ns)

        # aggregated
        self.total_ms: Dict[str, float] = {}
        self.count: Dict[str, int] = {}
        self.group_total_ms: Dict[str, float] = {}
        self.group_count: Dict[str, int] = {}

    def _group_key(self, module_name: str) -> Optional[str]:
        # Common transformer-wide ops outside per-layer blocks
        # (HuggingFace Llama): embed tokens and lm_head
        if "embed_tokens" in module_name:
            return "embed"
        if "lm_head" in module_name:
            return "lm_head"

        if self._group_re is None:
            return None
        m = self._group_re.search(module_name)
        if not m:
            return None
        # If regex has a capture group, use group(1). Otherwise use full match.
        return m.group(1) if m.lastindex else m.group(0)

    def _is_cuda(self) -> bool:
        return torch.cuda.is_available() and self.device_index is not None

    def set_current(self, *, phase: str, mb_idx: int) -> None:
        if not self.enabled:
            return
        self._current_phase = phase
        self._current_mb = mb_idx

    def clear_current(self) -> None:
        self._current_phase = None
        self._current_mb = None

    def should_profile_module(self, module_name: str, module: torch.nn.Module) -> bool:
        if self._class_name_allow is not None:
            return module.__class__.__name__ in self._class_name_allow
        if self._name_re is None:
            return False
        return bool(self._name_re.match(module_name))

    def attach(self, root_module: torch.nn.Module) -> None:
        """
        Attach hooks to modules under root_module that match should_profile_module(name).
        """
        if not self.enabled:
            return

        for name, mod in root_module.named_modules():
            if not self.should_profile_module(name, mod):
                continue

            def _pre_hook(_mod, _inputs, *, _name=name):
                if not self.enabled:
                    return
                if self._current_phase is None or self._current_mb is None:
                    return
                if self._current_phase != "fwd":
                    return

                key = (_name, self._current_phase, self._current_mb)
                if self._is_cuda():
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record()
                    self._inflight_cuda.setdefault(key, []).append(ev)
                else:
                    t0 = time.perf_counter_ns()
                    self._inflight_cpu.setdefault(key, []).append(t0)

            def _post_hook(_mod, _inputs, _output, *, _name=name):
                if not self.enabled:
                    return
                if self._current_phase is None or self._current_mb is None:
                    return
                if self._current_phase != "fwd":
                    return

                key = (_name, self._current_phase, self._current_mb)
                if self._is_cuda():
                    stack = self._inflight_cuda.get(key)
                    if not stack:
                        return
                    start_ev = stack.pop()
                    end_ev = torch.cuda.Event(enable_timing=True)
                    end_ev.record()
                    self._pending_cuda.append((_name, _CudaEventPair(start=start_ev, end=end_ev)))
                else:
                    stack = self._inflight_cpu.get(key)
                    if not stack:
                        return
                    start_ns = stack.pop()
                    end_ns = time.perf_counter_ns()
                    self._pending_cpu.append((_name, start_ns, end_ns))

            self._handles.append(mod.register_forward_pre_hook(_pre_hook))
            self._handles.append(mod.register_forward_hook(_post_hook))

    def detach(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def flush_current(self) -> None:
        """
        Convert pending events for the current micro-batch into aggregated totals.
        Call this AFTER finishing the forward of a micro-batch.
        """
        if not self.enabled:
            return

        if self._is_cuda():
            torch.cuda.synchronize(self.device_index)
            for name, pair in self._pending_cuda:
                ms = pair.start.elapsed_time(pair.end)
                self.total_ms[name] = self.total_ms.get(name, 0.0) + float(ms)
                self.count[name] = self.count.get(name, 0) + 1
                gk = self._group_key(name)
                if gk is not None:
                    self.group_total_ms[gk] = self.group_total_ms.get(gk, 0.0) + float(ms)
                    self.group_count[gk] = self.group_count.get(gk, 0) + 1
            self._pending_cuda.clear()
        else:
            for name, start_ns, end_ns in self._pending_cpu:
                ms = (end_ns - start_ns) / 1_000_000.0
                self.total_ms[name] = self.total_ms.get(name, 0.0) + float(ms)
                self.count[name] = self.count.get(name, 0) + 1
                gk = self._group_key(name)
                if gk is not None:
                    self.group_total_ms[gk] = self.group_total_ms.get(gk, 0.0) + float(ms)
                    self.group_count[gk] = self.group_count.get(gk, 0) + 1
            self._pending_cpu.clear()

    def reset_stats(self) -> None:
        self.total_ms.clear()
        self.count.clear()
        self.group_total_ms.clear()
        self.group_count.clear()

    def report_lines(self) -> List[str]:
        if not self.total_ms:
            return []

        def _layer_key(n: str) -> int:
            # Prefer the last integer in the module name (e.g., model_layers_12, model_layers_12_self_attn, etc.)
            m = re.findall(r"(\d+)", n)
            return int(m[-1]) if m else 10**9

        lines = []
        for name in sorted(self.total_ms.keys(), key=_layer_key):
            total = self.total_ms[name]
            cnt = self.count.get(name, 0)
            avg = total / cnt if cnt else 0.0
            lines.append(f"{name}: total={total:.3f}ms, count={cnt}, avg={avg:.3f}ms")
        return lines

    def report_group_lines(self) -> List[str]:
        """
        Grouped report (e.g., transformer block index).
        group_name_regex must be configured.
        """
        if not self.group_total_ms:
            return []

        def _group_sort_key(k: str):
            # embed first, numeric layers next, lm_head last, then others
            if k == "embed":
                return (0, -1)
            if k == "lm_head":
                return (2, 10**9)
            m = re.findall(r"(\d+)", k)
            if m:
                return (1, int(m[-1]))
            return (3, 10**9)

        lines = []
        for key in sorted(self.group_total_ms.keys(), key=_group_sort_key):
            total = self.group_total_ms[key]
            cnt = self.group_count.get(key, 0)
            avg = total / cnt if cnt else 0.0
            lines.append(f"layer[{key}]: total={total:.3f}ms, ops={cnt}, avg/op={avg:.3f}ms")
        return lines


