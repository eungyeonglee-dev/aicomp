# GBS 32 vs GBS 2 프로파일링 결과 비교 분석

## 로그 파일
- **GBS=32**: `results/20260115183537.log`
- **GBS=2**: `results/gbs2-mbs1-layer16/20260115183413.log`

---

## 설정 비교

| 설정 | GBS=32 | GBS=2 |
|------|--------|-------|
| Global Batch Size | 32 | 2 |
| Micro-batch Size | 1 | 1 |
| **num_mb (micro-batches per step)** | **32** | **2** |
| PP Size | 2 | 2 |

---

## 결과 비교

### Per-Layer 평균 시간

| 지표 | GBS=32 | GBS=2 | 차이 |
|------|--------|-------|------|
| Avg per layer (median) | 1.32 ms | 1.83 ms | +38% |
| Avg (excl 1st/last) median | 1.32 ms | 1.80 ms | +36% |

### Layer 0 (Stage 0 첫 번째 레이어)

| 지표 | GBS=32 | GBS=2 |
|------|--------|-------|
| Mean | 1.41 ms | 3.46 ms |
| Median | 1.34 ms | 2.42 ms |
| **Std** | **0.21** | **3.32** |
| Max | 1.99 ms | 13.41 ms |

### Layer 8 (Stage 1 첫 번째 레이어)

| 지표 | GBS=32 | GBS=2 |
|------|--------|-------|
| Mean | 1.99 ms | 8.88 ms |
| Median | 1.57 ms | 2.07 ms |
| **Max** | **5.87 ms** | **71.49 ms** |

---

## 차이 원인 분석

### 1. Pipeline Bubble 오버헤드

```
1F1B Pipeline Parallel 스케줄링:

GBS=2 (num_mb=2):
  Stage 0: [F0][F1]--------[B1][B0]
  Stage 1: ----[F0][F1][B1][B0]----
                 ↑ bubble 비율 높음

GBS=32 (num_mb=32):
  Stage 0: [F0][F1]...[F31]--------[B31]...[B0]
  Stage 1: ----[F0][F1]...[F31][B31]...[B0]----
                       ↑ steady-state 구간 길음
```

- **num_mb=2**: 파이프라인 startup/shutdown 오버헤드가 각 forward에 더 많이 분배됨
- **num_mb=32**: steady-state 구간이 길어서 오버헤드가 희석됨

### 2. PP 통신 대기 시간 포함

Stage 경계 레이어 (layer_0, layer_8)에서 PP 통신 대기 시간이 측정에 포함됨:

| Layer | GBS=32 Max | GBS=2 Max | 비고 |
|-------|------------|-----------|------|
| layer_0 | 1.99 ms | 13.41 ms | Stage 0 시작 |
| layer_8 | 5.87 ms | **71.49 ms** | Stage 1 시작 (PP recv 대기) |

- **num_mb가 작으면**: Stage 1이 Stage 0의 출력을 기다리는 시간이 상대적으로 크게 측정됨
- **num_mb가 크면**: 파이프라인이 "채워진" 상태에서 대기 시간이 줄어듦

### 3. 측정 방식에 의한 평균화 효과

```python
# LayerBlockProfiler의 측정 방식
per_forward_time = total_step_accumulated_time / num_mb
```

| num_mb | 설명 |
|--------|------|
| 32 | 32번의 forward 시간을 합산 후 32로 나눔 → 평균화 효과 큼 |
| 2 | 2번의 forward만 합산 → 분산이 큼 (outlier 영향 큼) |

**표준편차 비교**:
- GBS=32 layer_0 Std: 0.21 (안정적)
- GBS=2 layer_0 Std: 3.32 (불안정)

### 4. CUDA 커널 스케줄링

- num_mb가 클수록 CUDA 스트림에서 연속적인 커널 실행이 가능
- 커널 launch overhead가 amortize됨

---

## 결론

| 원인 | 영향 |
|------|------|
| **Pipeline bubble** | num_mb 작으면 bubble 비율↑ → per-forward 시간↑ |
| **PP 통신 대기** | Stage 경계 레이어에서 recv 대기 시간이 측정에 포함 |
| **통계적 평균화** | num_mb 클수록 평균이 안정적, outlier 영향 감소 |
| **CUDA 스케줄링** | num_mb 클수록 커널 실행 효율↑ |

---

## 권장 사항

### 정확한 레이어 연산 시간 측정을 위해:

1. **GBS를 충분히 크게 설정** (num_mb >= 16 권장)
   - Pipeline steady-state 구간 확보
   - 통계적 평균화 효과

2. **Median 값 사용**
   - Mean보다 outlier에 강건함
   - GBS=2에서도 median은 상대적으로 안정적

3. **Stage 경계 레이어 분리 분석**
   - layer_0, layer_8 (PP stage 시작점)은 별도 분석
   - PP 통신 대기 시간 포함 가능성

4. **`Avg (excl 1st/last)` 참고**
   - 첫 번째/마지막 레이어 제외한 평균
   - 가장 안정적인 레이어 시간 추정치

---

## 실제 레이어 연산 시간 추정

| 조건 | 추정 방법 | 값 |
|------|----------|-----|
| GBS=32 | Avg (excl 1st/last) median | **~1.32 ms** |
| GBS=2 | Avg (excl 1st/last) median | ~1.80 ms (신뢰도 낮음) |

**결론**: GBS=32의 결과가 실제 레이어 연산 시간에 더 가깝습니다.
