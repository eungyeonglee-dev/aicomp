import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.distributed as dist

try:
    # Used by TP/DTensor/functional collectives
    from torch.distributed._functional_collectives import AsyncCollectiveTensor  # type: ignore
    import torch.distributed._functional_collectives as _fc  # type: ignore
except Exception:  # pragma: no cover
    AsyncCollectiveTensor = None  # type: ignore
    _fc = None  # type: ignore

try:
    from torch.distributed.tensor import DTensor  # type: ignore
except Exception:  # pragma: no cover
    DTensor = None  # type: ignore


def _force_async_collective_wait(output: Any) -> Any:
    """
    Force DTensor/AsyncCollectiveTensor async collectives to complete.
    This ensures TP all-reduce is included in layer timing measurements.
    """
    if output is None:
        return output

    # Handle AsyncCollectiveTensor directly
    if AsyncCollectiveTensor is not None and isinstance(output, AsyncCollectiveTensor):
        return output.wait()

    # Handle DTensor (wraps AsyncCollectiveTensor internally)
    if DTensor is not None and isinstance(output, DTensor):
        # DTensor._local_tensor may be AsyncCollectiveTensor
        local = output._local_tensor
        if AsyncCollectiveTensor is not None and isinstance(local, AsyncCollectiveTensor):
            local.wait()
        return output

    # Handle tuple/list of outputs (common in transformer blocks)
    if isinstance(output, (tuple, list)):
        result = [_force_async_collective_wait(o) for o in output]
        return type(output)(result)

    # Handle dict outputs
    if isinstance(output, dict):
        return {k: _force_async_collective_wait(v) for k, v in output.items()}

    # Regular tensor or other type - no action needed
    return output


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
        self._module_stack: List[str] = []

        self._handles = []

        # inflight stacks: key -> list[start_marker]
        self._inflight_cuda: Dict[Tuple[str, str, int], List[torch.cuda.Event]] = {}
        self._inflight_cpu: Dict[Tuple[str, str, int], List[int]] = {}

        # pending completed measurements for current context
        self._pending_cuda: List[Tuple[str, _CudaEventPair]] = []
        self._pending_cpu: List[Tuple[str, int, int]] = []  # (name, start_ns, end_ns)

        # pending communication kernel timings (CUDA events)
        self._pending_comm_cuda: List[Tuple[str, _CudaEventPair]] = []
        self._pending_comm_cpu: List[Tuple[str, int, int]] = []

        # aggregated
        self.total_ms: Dict[str, float] = {}
        self.count: Dict[str, int] = {}
        self.group_total_ms: Dict[str, float] = {}
        self.group_count: Dict[str, int] = {}

        # communication (collective) time attributed to current module stack top
        self.comm_total_ms: Dict[str, float] = {}
        self.comm_count: Dict[str, int] = {}
        self.comm_group_total_ms: Dict[str, float] = {}
        self.comm_group_count: Dict[str, int] = {}

    def _current_module_name(self) -> Optional[str]:
        return self._module_stack[-1] if self._module_stack else None

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
        _set_active_profiler(self)

    def clear_current(self) -> None:
        self._current_phase = None
        self._current_mb = None
        _clear_active_profiler(self)

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
                self._module_stack.append(_name)

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

                # Force DTensor async collectives (all-reduce) to complete before measuring end time.
                # This ensures TP communication time is included in the layer timing.
                _output = _force_async_collective_wait(_output)

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

                # pop module stack (best-effort)
                if self._module_stack:
                    self._module_stack.pop()

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

            # collectives
            for name, pair in self._pending_comm_cuda:
                ms = pair.start.elapsed_time(pair.end)
                self._record_comm_ms_for(name, float(ms))
            self._pending_comm_cuda.clear()
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

            for name, start_ns, end_ns in self._pending_comm_cpu:
                ms = (end_ns - start_ns) / 1_000_000.0
                self._record_comm_ms_for(name, float(ms))
            self._pending_comm_cpu.clear()

    def reset_stats(self) -> None:
        self.total_ms.clear()
        self.count.clear()
        self.group_total_ms.clear()
        self.group_count.clear()
        self.comm_total_ms.clear()
        self.comm_count.clear()
        self.comm_group_total_ms.clear()
        self.comm_group_count.clear()
        self._pending_comm_cuda.clear()
        self._pending_comm_cpu.clear()

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
            compute_ms = self.group_total_ms.get(key, 0.0)
            compute_ops = self.group_count.get(key, 0)
            comm_ms = self.comm_group_total_ms.get(key, 0.0)
            comm_ops = self.comm_group_count.get(key, 0)
            total_ms = compute_ms + comm_ms
            # Calculate average per forward pass (compute_ops is the number of module forward calls)
            avg_compute_ms = compute_ms / compute_ops if compute_ops > 0 else 0.0
            avg_comm_ms = comm_ms / comm_ops if comm_ops > 0 else 0.0
            avg_total_ms = total_ms / compute_ops if compute_ops > 0 else 0.0
            lines.append(
                f"layer[{key}]: compute={compute_ms:.3f}ms (avg={avg_compute_ms:.3f}ms), "
                f"comm={comm_ms:.3f}ms (avg={avg_comm_ms:.3f}ms), "
                f"total={total_ms:.3f}ms (avg={avg_total_ms:.3f}ms), ops={compute_ops}"
            )
        return lines

    def get_avg_times_dict(self) -> Dict[str, float]:
        """
        Get average times (compute + comm) per layer as a dictionary.
        Keys: 'embed', '0', '1', ..., 'lm_head'
        Values: average total time in ms per forward pass
        """
        result = {}
        for key in self.group_total_ms.keys():
            compute_ms = self.group_total_ms.get(key, 0.0)
            compute_ops = self.group_count.get(key, 0)
            comm_ms = self.comm_group_total_ms.get(key, 0.0)
            total_ms = compute_ms + comm_ms
            avg_total_ms = total_ms / compute_ops if compute_ops > 0 else 0.0
            result[key] = avg_total_ms
        return result

    def save_profile_numpy(
        self,
        model_name: str,
        gpu_type: str,
        tp_size: int,
        num_layers: int,
        output_dir: str = ".",
        add_name: Optional[str] = None,
    ) -> str:
        """
        Save layer profile as numpy array file.

        Format: [embed_time, layer0_time, layer1_time, ..., layerN_time, lm_head_time]
        Each time is the average total time (compute + comm) per forward pass in ms.

        Args:
            model_name: Model name (e.g., 'llama-3.2-1b')
            gpu_type: GPU type (e.g., 'A100', 'H100')
            tp_size: Tensor parallelism size
            num_layers: Total number of transformer layers in the model
            output_dir: Directory to save the numpy file
            add_name: Additional name suffix for the file

        Returns:
            Path to the saved numpy file
        """
        import numpy as np
        import os

        avg_times = self.get_avg_times_dict()

        # Build profile array: [embed, layer0, layer1, ..., layerN-1, lm_head]
        profile_array = []

        # Embed time
        embed_time = avg_times.get("embed", 0.0)
        profile_array.append(embed_time)

        # Layer times (0 to num_layers-1)
        for i in range(num_layers):
            layer_time = avg_times.get(str(i), 0.0)
            profile_array.append(layer_time)

        # lm_head time
        lm_head_time = avg_times.get("lm_head", 0.0)
        profile_array.append(lm_head_time)

        # Create filename
        add_suffix = f"_{add_name}" if add_name else ""
        filename = f"{model_name}_{gpu_type}_tp{tp_size}{add_suffix}.npy"
        filepath = os.path.join(output_dir, filename)

        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save numpy array
        np.save(filepath, np.array(profile_array, dtype=np.float32))

        return filepath

    def enable_collective_profiling(self) -> None:
        """
        Patch common collectives so TP all-reduce/all-gather/reduce-scatter wait time is attributed
        to the currently executing module (top of the module stack).
        """
        if not self.enabled:
            return
        _patch_collectives()

    def _record_comm_ms(self, ms: float) -> None:
        if not self.enabled:
            return
        if self._current_phase != "fwd":
            return
        name = self._current_module_name() or "__no_module__"
        self._record_comm_ms_for(name, ms)

    def _record_comm_ms_for(self, name: str, ms: float) -> None:
        if not self.enabled:
            return
        if self._current_phase != "fwd":
            return
        self.comm_total_ms[name] = self.comm_total_ms.get(name, 0.0) + float(ms)
        self.comm_count[name] = self.comm_count.get(name, 0) + 1
        # If we couldn't attribute to a module, still expose comm under a stable group key.
        gk = self._group_key(name) or ("tp_comm" if name == "__no_module__" else None)
        if gk is not None:
            self.comm_group_total_ms[gk] = self.comm_group_total_ms.get(gk, 0.0) + float(ms)
            self.comm_group_count[gk] = self.comm_group_count.get(gk, 0) + 1


# ---- global patching for collectives (best-effort, process-local) -----------------

_ACTIVE_PROFILER: Optional[LayerTimeProfiler] = None
_COLLECTIVES_PATCHED = False
_ORIG_ASYNC_WAIT: Optional[Callable[..., Any]] = None
_ORIG_DIST_ALL_REDUCE: Optional[Callable[..., Any]] = None
_ORIG_FC_WAIT_TENSOR: Optional[Callable[..., Any]] = None
_ORIG_C10D_WAIT_TENSOR: Optional[Callable[..., Any]] = None
_ORIG_C10D_ALL_REDUCE: Optional[Callable[..., Any]] = None
_ORIG_C10D_ALL_GATHER_INTO_TENSOR: Optional[Callable[..., Any]] = None
_ORIG_C10D_REDUCE_SCATTER_TENSOR: Optional[Callable[..., Any]] = None


def _set_active_profiler(p: LayerTimeProfiler) -> None:
    global _ACTIVE_PROFILER
    _ACTIVE_PROFILER = p


def _clear_active_profiler(p: LayerTimeProfiler) -> None:
    global _ACTIVE_PROFILER
    if _ACTIVE_PROFILER is p:
        _ACTIVE_PROFILER = None


def _patch_collectives() -> None:
    global _COLLECTIVES_PATCHED
    global _ORIG_ASYNC_WAIT, _ORIG_DIST_ALL_REDUCE, _ORIG_FC_WAIT_TENSOR, _ORIG_C10D_WAIT_TENSOR
    global _ORIG_C10D_ALL_REDUCE, _ORIG_C10D_ALL_GATHER_INTO_TENSOR, _ORIG_C10D_REDUCE_SCATTER_TENSOR
    if _COLLECTIVES_PATCHED:
        return

    # Patch AsyncCollectiveTensor.wait (TP/DTensor path)
    if AsyncCollectiveTensor is not None and hasattr(AsyncCollectiveTensor, "wait"):
        _ORIG_ASYNC_WAIT = AsyncCollectiveTensor.wait  # type: ignore[attr-defined]

        def _wait_wrapped(self, *args, **kwargs):  # type: ignore[no-redef]
            t0 = time.perf_counter_ns()
            out = _ORIG_ASYNC_WAIT(self, *args, **kwargs)  # type: ignore[misc]
            t1 = time.perf_counter_ns()
            ms = (t1 - t0) / 1_000_000.0
            p = _ACTIVE_PROFILER
            if p is not None:
                p._record_comm_ms(ms)
            return out

        AsyncCollectiveTensor.wait = _wait_wrapped  # type: ignore[assignment]

    # Patch torch.distributed._functional_collectives.wait_tensor (common TP path)
    if _fc is not None and hasattr(_fc, "wait_tensor"):
        _ORIG_FC_WAIT_TENSOR = _fc.wait_tensor  # type: ignore[attr-defined]

        def _fc_wait_tensor_wrapped(*args, **kwargs):  # type: ignore[no-redef]
            t0 = time.perf_counter_ns()
            out = _ORIG_FC_WAIT_TENSOR(*args, **kwargs)  # type: ignore[misc]
            t1 = time.perf_counter_ns()
            ms = (t1 - t0) / 1_000_000.0
            p = _ACTIVE_PROFILER
            if p is not None:
                p._record_comm_ms(ms)
            return out

        _fc.wait_tensor = _fc_wait_tensor_wrapped  # type: ignore[assignment]

    # Patch torch.ops._c10d_functional.wait_tensor (lower-level functional op)
    try:
        ns = torch.ops._c10d_functional
        if hasattr(ns, "wait_tensor"):
            _ORIG_C10D_WAIT_TENSOR = ns.wait_tensor

            def _c10d_wait_tensor_wrapped(*args, **kwargs):  # type: ignore[no-redef]
                t0 = time.perf_counter_ns()
                out = _ORIG_C10D_WAIT_TENSOR(*args, **kwargs)  # type: ignore[misc]
                t1 = time.perf_counter_ns()
                ms = (t1 - t0) / 1_000_000.0
                p = _ACTIVE_PROFILER
                if p is not None:
                    p._record_comm_ms(ms)
                return out

            ns.wait_tensor = _c10d_wait_tensor_wrapped  # type: ignore[assignment]

        # Patch collective *launch* ops to capture GPU kernel time even when there is no explicit wait_tensor().
        if hasattr(ns, "all_reduce"):
            _ORIG_C10D_ALL_REDUCE = ns.all_reduce

            def _c10d_all_reduce_wrapped(*args, **kwargs):  # type: ignore[no-redef]
                p = _ACTIVE_PROFILER
                if p is None or not p.enabled:
                    return _ORIG_C10D_ALL_REDUCE(*args, **kwargs)  # type: ignore[misc]
                name = p._current_module_name() or "__no_module__"
                if p._is_cuda():
                    start_ev = torch.cuda.Event(enable_timing=True)
                    end_ev = torch.cuda.Event(enable_timing=True)
                    start_ev.record()
                    out = _ORIG_C10D_ALL_REDUCE(*args, **kwargs)  # type: ignore[misc]
                    end_ev.record()
                    p._pending_comm_cuda.append((name, _CudaEventPair(start=start_ev, end=end_ev)))
                    return out
                t0 = time.perf_counter_ns()
                out = _ORIG_C10D_ALL_REDUCE(*args, **kwargs)  # type: ignore[misc]
                t1 = time.perf_counter_ns()
                p._pending_comm_cpu.append((name, t0, t1))
                return out

            ns.all_reduce = _c10d_all_reduce_wrapped  # type: ignore[assignment]

        if hasattr(ns, "all_gather_into_tensor"):
            _ORIG_C10D_ALL_GATHER_INTO_TENSOR = ns.all_gather_into_tensor

            def _c10d_all_gather_into_tensor_wrapped(*args, **kwargs):  # type: ignore[no-redef]
                p = _ACTIVE_PROFILER
                if p is None or not p.enabled:
                    return _ORIG_C10D_ALL_GATHER_INTO_TENSOR(*args, **kwargs)  # type: ignore[misc]
                name = p._current_module_name() or "__no_module__"
                if p._is_cuda():
                    start_ev = torch.cuda.Event(enable_timing=True)
                    end_ev = torch.cuda.Event(enable_timing=True)
                    start_ev.record()
                    out = _ORIG_C10D_ALL_GATHER_INTO_TENSOR(*args, **kwargs)  # type: ignore[misc]
                    end_ev.record()
                    p._pending_comm_cuda.append((name, _CudaEventPair(start=start_ev, end=end_ev)))
                    return out
                t0 = time.perf_counter_ns()
                out = _ORIG_C10D_ALL_GATHER_INTO_TENSOR(*args, **kwargs)  # type: ignore[misc]
                t1 = time.perf_counter_ns()
                p._pending_comm_cpu.append((name, t0, t1))
                return out

            ns.all_gather_into_tensor = _c10d_all_gather_into_tensor_wrapped  # type: ignore[assignment]

        if hasattr(ns, "reduce_scatter_tensor"):
            _ORIG_C10D_REDUCE_SCATTER_TENSOR = ns.reduce_scatter_tensor

            def _c10d_reduce_scatter_tensor_wrapped(*args, **kwargs):  # type: ignore[no-redef]
                p = _ACTIVE_PROFILER
                if p is None or not p.enabled:
                    return _ORIG_C10D_REDUCE_SCATTER_TENSOR(*args, **kwargs)  # type: ignore[misc]
                name = p._current_module_name() or "__no_module__"
                if p._is_cuda():
                    start_ev = torch.cuda.Event(enable_timing=True)
                    end_ev = torch.cuda.Event(enable_timing=True)
                    start_ev.record()
                    out = _ORIG_C10D_REDUCE_SCATTER_TENSOR(*args, **kwargs)  # type: ignore[misc]
                    end_ev.record()
                    p._pending_comm_cuda.append((name, _CudaEventPair(start=start_ev, end=end_ev)))
                    return out
                t0 = time.perf_counter_ns()
                out = _ORIG_C10D_REDUCE_SCATTER_TENSOR(*args, **kwargs)  # type: ignore[misc]
                t1 = time.perf_counter_ns()
                p._pending_comm_cpu.append((name, t0, t1))
                return out

            ns.reduce_scatter_tensor = _c10d_reduce_scatter_tensor_wrapped  # type: ignore[assignment]
    except Exception:
        pass

    # Patch dist.all_reduce (some TP/DP paths may call this directly; count only if profiler active)
    if hasattr(dist, "all_reduce"):
        _ORIG_DIST_ALL_REDUCE = dist.all_reduce

        def _all_reduce_wrapped(*args, **kwargs):  # type: ignore[no-redef]
            t0 = time.perf_counter_ns()
            out = _ORIG_DIST_ALL_REDUCE(*args, **kwargs)  # type: ignore[misc]
            t1 = time.perf_counter_ns()
            ms = (t1 - t0) / 1_000_000.0
            p = _ACTIVE_PROFILER
            # Only attribute if we're inside a profiled forward-module context.
            if p is not None and p._current_module_name() is not None:
                p._record_comm_ms(ms)
            return out

        dist.all_reduce = _all_reduce_wrapped  # type: ignore[assignment]

    _COLLECTIVES_PATCHED = True


