# Layer Profiling 분석 결과

## 로그 파일
- `results/20260115182108.log`

## 분석 대상
- L3274: `layer_0` - 3.6466ms (과도하게 큼)
- L3321: `layer_15` - 1.5815ms (마지막 레이어인데 작음)

---

## 1. Layer 0이 과도하게 큰 이유 (3.6ms vs ~1.7ms)

### 원인: 측정 범위 + CUDA warm-up 잔여 효과

측정 방식:
- **Layer N**: `q_proj_N start → q_proj_(N+1) start`
- Layer 0은 `q_proj_0 start`부터 시작

실제 FX 그래프 구조:
```
embed_tokens → [input_layernorm_0] → q_proj_0 → ... → [input_layernorm_1] → q_proj_1
```

### Layer 0 측정 범위
- `q_proj_0` ~ `q_proj_1` 사이에는 **layer_1의 input_layernorm**이 포함됨

### 왜 크게 나오나?

| 지표 | 값 |
|------|-----|
| Mean | 3.6466ms |
| Median | 2.5196ms |
| Min | 1.7909ms |
| Max | 15.4045ms |
| **Std** | **3.9368ms** |

- 표준편차가 매우 큼 (Max=15.4ms, Min=1.7ms)
- **첫 번째 레이어의 CUDA 커널 warm-up 오버헤드**가 아직 남아있음
- **PP stage 경계에서 통신 대기** 시간이 포함되었을 가능성

---

## 2. 마지막 Layer (layer_15)가 작은 이유

### 로그 확인 (L3234)
```
layer_7: q_proj start → lm_head start (includes model_norm)
```

### 마지막 레이어 측정 범위
- `q_proj_last start → lm_head start`
- `model_norm`이 **포함**된다고 명시됨

### 실제 측정값 비교

| Layer | Median(ms) | 비고 |
|-------|------------|------|
| layer_14 | 1.6916 | |
| layer_15 | 1.5811 | 마지막 레이어 |

실제로 크게 다르지 않음. layer_15가 약간 작은 이유:

1. **PP Stage 분할**: layer_15는 Stage 1 (rank 1)의 마지막 레이어
   - Stage 1: layer_8 ~ layer_15 + lm_head
   - `lm_head start` 시점에서 측정이 끝나므로, 다음 레이어로 넘어가는 오버헤드 없음

2. **다음 input_layernorm 없음**: 마지막 레이어는 다음 레이어의 `input_layernorm`을 포함하지 않음 (model_norm만 포함)

---

## 요약

| 현상 | 원인 |
|------|------|
| Layer 0이 큼 | CUDA warm-up 잔여 + PP 통신 대기 + 높은 분산 |
| 마지막 layer가 작음 | 다음 레이어의 input_layernorm 미포함 + PP stage 경계 효과 |

---

## 권장 사항

1. **더 정확한 측정을 위해**: warmup step을 더 늘리기 (현재 50 → 100 이상)
2. **안정적인 값 참고**: Mean 대신 **Median** 값 사용 권장
   - layer_0: Mean=3.6466ms vs **Median=2.5196ms** (더 안정적)
3. **Layer 0 분산이 큰 경우**: 첫 번째 측정 스텝을 추가로 제외하는 로직 고려
