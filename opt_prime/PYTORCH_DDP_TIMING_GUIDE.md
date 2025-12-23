# PyTorch DDP All-Reduce 통신 시간 측정 가이드

## 방법 1: Hook 사용 (권장)

현재 제공된 `ddp_timing.py` 모듈을 사용하여 DDP의 reducer에 hook을 설치합니다.

### 사용 방법

1. `opti_pri.py`에서 DDP를 생성한 후 hook 설치:

```python
from opt_prime.ddp_timing import install_ddp_reducer_hook

# DDP 생성 후
self.run_info.submod = DistributedDataParallel(...)
# Hook 설치
install_ddp_reducer_hook(self.timers)
```

2. `schedule.py`에서 이미 사용 중인 타이머로 자동 측정됩니다.

## 방법 2: PyTorch 소스 코드 직접 수정

PyTorch 소스 코드를 직접 수정하여 all-reduce 통신 시간만 측정하는 방법입니다.

### 수정할 파일 위치

PyTorch 설치 경로에서 다음 파일을 수정:
- `torch/nn/parallel/distributed.py` (또는 `torch/distributed/nn/api.py`)

### 수정 방법

#### Option A: `_all_reduce_bucket` 메서드 수정

`torch/nn/parallel/distributed.py` 파일에서 `_Reducer` 클래스의 `_all_reduce_bucket` 메서드를 찾아 수정:

```python
# 기존 코드 (약 800-900줄 근처)
def _all_reduce_bucket(self, bucket):
    # ... 기존 코드 ...
    bucket.future_result = dist.all_reduce(bucket.buffer, async_op=True, group=self.process_group)
    # ... 기존 코드 ...
```

수정 후:

```python
def _all_reduce_bucket(self, bucket):
    # 타이머 시작 (전역 변수로 timers 객체가 설정되어 있다고 가정)
    import time
    start_time = time.time()
    # 또는 CUDA event 사용
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    
    # ... 기존 코드 ...
    bucket.future_result = dist.all_reduce(bucket.buffer, async_op=True, group=self.process_group)
    
    # all-reduce 완료 대기
    bucket.future_result.wait()
    
    # 타이머 종료
    if torch.cuda.is_available():
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        # 로깅 (전역 변수 또는 환경 변수 사용)
        if hasattr(self, '_timing_log'):
            self._timing_log.append(elapsed_ms)
    else:
        elapsed_time = time.time() - start_time
        if hasattr(self, '_timing_log'):
            self._timing_log.append(elapsed_time * 1000)
    
    # ... 기존 코드 ...
```

#### Option B: `dist.all_reduce` 래핑

`torch/distributed/nccl.py` 또는 `torch/distributed/c10d.py`에서 `all_reduce` 함수를 수정:

```python
def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    # 타이밍 시작
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    
    # 기존 all_reduce 호출
    result = _original_all_reduce(tensor, op=op, group=group, async_op=async_op)
    
    # 완료 대기
    if async_op:
        result.wait()
    else:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # 타이밍 종료 및 로깅
    if torch.cuda.is_available():
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        # 로깅 처리
        _log_all_reduce_time(elapsed_ms)
    
    return result
```

### PyTorch 소스 코드 위치 찾기

```bash
# Python에서 PyTorch 설치 경로 확인
python -c "import torch; print(torch.__file__)"

# 일반적인 위치:
# - Conda: ~/anaconda3/lib/python3.x/site-packages/torch/
# - Pip: /usr/local/lib/python3.x/site-packages/torch/
```

### 주의사항

1. **PyTorch 버전 호환성**: PyTorch 버전이 바뀌면 내부 구현이 변경될 수 있습니다.
2. **재컴파일 필요**: 소스 코드를 수정한 경우 PyTorch를 재설치해야 할 수 있습니다.
3. **유지보수**: PyTorch 업데이트 시 수정 사항을 다시 적용해야 합니다.

## 방법 3: 현재 구현된 Hook 사용 (가장 간단)

`ddp_timing.py`를 사용하면 PyTorch 소스 코드 수정 없이 측정 가능합니다.

### 설치 및 사용

```python
# opti_pri.py 또는 schedule.py에서
from opt_prime.ddp_timing import install_ddp_reducer_hook

# DDP 생성 후
install_ddp_reducer_hook(timers)
```

이 방법이 가장 안전하고 유지보수가 쉽습니다.












