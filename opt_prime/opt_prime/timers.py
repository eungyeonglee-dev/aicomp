from abc import ABC
from abc import abstractmethod
import time
import os
from typing import Any
import torch

_GLOBAL_TIMERS = None

def _set_timers(log_level):
    global _GLOBAL_TIMERS
    _GLOBAL_TIMERS = Timers(log_level=log_level)

def get_timers(log_level):
    _set_timers(log_level)
    return _GLOBAL_TIMERS

    

class TimerClass(ABC):
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def start(self, barrier=False):
        pass
    
    @abstractmethod
    def stop(self, barrier=False):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def elapsed(self, reset=True, barrier=False):
        pass
        
class DummyTimer(TimerClass):
    
    def __init__(self):
        super().__init__('dummy timer')
    
    def start(self, barrier=False):
        return
    
    def stop(self, barrier=False):
        return
    
    def reset(self):
        return
    
    def elapsed(self, reset=True, barrier=False):
        # Return 0.0 instead of None to ensure tensor assignment works correctly
        # This prevents issues when gathering timer data across ranks
        return 0.0

class cudaTimer(TimerClass):
    def __init__(self, name):
        super().__init__(name)
        self._startevent = torch.cuda.Event(enable_timing=True)
        self._endevent = torch.cuda.Event(enable_timing=True)
        self._elapsed = 0.0
        self._started = False
    
    def start(self, barrier=False):
        self._startevent.record()
        self._started = True
    
    def stop(self, barrier=False):
        assert self._started, "Timer is not started."
        
        self._endevent.record()
        torch.cuda.synchronize()
        # elapsed_time() returns milliseconds, convert to seconds and round to 3 decimal places
        elapsed_ms = self._startevent.elapsed_time(self._endevent)
        elapsed_sec = round(elapsed_ms / 1000.0, 3)
        self._elapsed += elapsed_sec
        self._started = False
        
        
    def reset(self):
        self._elapsed = 0.0
        self._started = False

    def elapsed(self, reset=True, barrier=False):
        _started = self._started
        # 시간 재기 멈춤이 안된 상황. 시간을 재는 중이기 때문에 멈춤을 해줌.
        if self._started:
            self.stop(barrier=barrier)
        # 시간재기가 완료 되었다면, 그 시간을 얻고, 시간을 재는 중이었다면 stop함수로 시간을 얻음
        _elapsed = self._elapsed
        
        # 시간을 잴 때마다 새로 재는 거라면
        if reset:
            self.reset()
        
        # 시간을 재고 있었던 거라서 stop()했기 때문에 다시 시간을 재기 시작함
        if _started:
            self.start(barrier=barrier)
        return _elapsed
        
        
class Timers:
    
    def __init__(self, log_level):
        self._log_level = log_level
        self._timers = {}
        self._log_levels = {}
        self._max_log_level = 2
        self._dummy_timer = DummyTimer()  # Create an instance, not the class
        # Lazy initialization: get rank and world_size when needed
        # Try torch.distributed first, fall back to environment variables if not initialized
        self._world_size = None
        self._rank = None
        
    
    def _get_world_size(self):
        """Get world_size, with fallback to environment variable if distributed not initialized."""
        if self._world_size is None:
            if torch.distributed.is_initialized():
                self._world_size = torch.distributed.get_world_size()
            else:
                # Fallback to environment variable (set by torchrun)
                self._world_size = int(os.environ.get('WORLD_SIZE', 1))
        return self._world_size
    
    def _get_rank(self):
        """Get rank, with fallback to environment variable if distributed not initialized."""
        if self._rank is None:
            if torch.distributed.is_initialized():
                self._rank = torch.distributed.get_rank()
            else:
                # Fallback to environment variable (set by torchrun)
                self._rank = int(os.environ.get('RANK', 0))
        return self._rank
    
    def __call__(self, name, log_level=None):
        if name in self._timers:
            return self._timers[name]
        
        if log_level is None:
            log_level = self._max_log_level
        
        if log_level > self._log_level:
            return self._dummy_timer
        
        self._timers[name] = cudaTimer(name)
        self._log_levels[name] = log_level
        return self._timers[name]
    
    def register_timer(self, name, log_level=2):
        """Register a timer without starting it. Useful for ensuring timer exists for gather operations."""
        if name not in self._timers:
            if log_level <= self._log_level:
                self._timers[name] = cudaTimer(name)
                self._log_levels[name] = log_level
    
    def get_timer_names(self):
        """Return a list of all registered timer names from all ranks."""
        # First, get local timer names
        local_timer_names = list(self._timers.keys())
        
        # If distributed is not initialized, just return local names
        if not torch.distributed.is_initialized():
            return local_timer_names
        
        # Gather all timer names from all ranks
        world_size = self._get_world_size()
        rank = self._get_rank()
        
        # Convert timer names to a set for union operation
        all_timer_names = set(local_timer_names)
        
        # Get current CUDA device
        current_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        
        # For each rank, we need to gather the timer names
        # Since we can't directly gather strings, we'll use a different approach:
        # Collect all unique timer names by checking if any rank has each timer
        # We'll iterate through a reasonable range of possible timer names
        # OR: use allgather to collect all timer names
        
        # Simpler approach: gather all timer names using allgather_object
        if hasattr(torch.distributed, 'all_gather_object'):
            # PyTorch 1.8+ has all_gather_object
            gathered_names = [None] * world_size
            torch.distributed.all_gather_object(gathered_names, local_timer_names)
            for names in gathered_names:
                all_timer_names.update(names)
        else:
            # Fallback: just return local names if all_gather_object is not available
            # This is a limitation but should work for most cases
            pass
        
        return sorted(list(all_timer_names))

    def _get_elapsed_time(self, names, reset):
        # Get rank and world_size (with lazy initialization)
        world_size = self._get_world_size()
        rank = self._get_rank()
        
        # Get current CUDA device (e.g., 'cuda:0', 'cuda:1', etc.)
        current_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        rank_name_to_time = torch.zeros((world_size, len(names)),
                            dtype=torch.float,
                            device=current_device)

        for i, name in enumerate(names):
            if name in self._timers:
                elapsed_time = self._timers[name].elapsed(reset=reset)
                # Ensure we have a valid float value (handle None from DummyTimer)
                if elapsed_time is None:
                    elapsed_time = 0.0
                rank_name_to_time[rank, i] = elapsed_time
            else:
                # Timer not registered in this rank, use 0.0
                rank_name_to_time[rank, i] = 0.0
        
        # Gather all ranks' data to rank 0 (first stage)
        # Changed from rank N-1 to rank 0 because:
        # - In pipeline parallelism warm-up phase, first stage ranks (rank 0~K) start working
        # - Last stage rank (rank N-1) may not participate in warm-up
        # - Using rank 0 as dst_rank ensures gather works even during warm-up
        # PyTorch's gather API requires the destination rank to provide a gather_list
        # gather_list : It's a list of tensors that each rank will send to rank 0
        dst_rank = 0
        if rank == dst_rank:
            # rank 0: use each row of rank_name_to_time as gather_list
            # the data is automatically stored in rank_name_to_time after gather
            # the size of gather_list is N which is the world size
            ##### gather_list #####
            # [rank_name_to_time[0, :], rank_name_to_time[1, :], ..., rank_name_to_time[N-1, :]]
            # rank_name_to_time[r, :] is the row of rank_name_to_time for rank r
            gather_list = [rank_name_to_time[r, :] for r in range(world_size)]
        else:
            # rank1 ~ rank N-1: send None (only send data, no receive)
            gather_list = None
        
        # torch.distributed.gather: each rank sends its row (rank_name_to_time[rank, :])
        # Ensure all ranks are ready before gather to prevent hanging
        # This is especially important during warm-up phase in pipeline parallelism
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        torch.distributed.gather(
            rank_name_to_time[rank, :],  # the data each rank sends
            gather_list=gather_list,           # the space the destination receives (only dst_rank is not None)
            dst=dst_rank                       # the destination rank number (rank 0)
        )
        
        return rank_name_to_time

    def _get_all_ranks_time_string(self, names, reset):
        if self._log_level == 0:
            return ""
        rank_name_to_time = self._get_elapsed_time(names, reset)
        world_size = self._get_world_size()
        output_string = 'times(ms)\n'
        
        # Access GPU tensor directly without moving to CPU (avoid bottleneck)
        # Format: rank0\nname1:xxx\nname2:xx (3 decimal places)
        # Define timers to include in steptime calculation (to avoid duplicates)
        steptime_timer_names = ['tokenizer','IR-preparing','prepare-labels','free-memory','forward-backward','optimizer-step']
        
        for rank in range(world_size):
            # Check if this rank has any timer data
            has_data = any(rank_name_to_time[rank, i].item() != 0.0 for i in range(len(names)))
            if has_data:
                output_string += f'==========rank{rank}==========\n'
                # Calculate sum of only specified timer values for this rank (to avoid duplicates)
                steptime_sum = 0.0
                for i, name in enumerate(names):
                    elapsed_val = rank_name_to_time[rank, i].item()
                    # Format to 3 decimal places
                    output_string += f'{name}:{elapsed_val:.3f}\n'
                    # Only include specified timers in steptime calculation
                    if name in steptime_timer_names:
                        steptime_sum += elapsed_val
                # Add steptime summary at the end
                output_string += f'rank{rank}_steptime:{steptime_sum:.3f}\n'
        
        return output_string

    def log(self, names):
        output_string = self._get_all_ranks_time_string(names, reset=True)
        rank = self._get_rank()
        # Print only on rank 0 (where gather collects all data)
        if rank == 0:
            print(output_string, flush=True)
        

        

    
    
            
        