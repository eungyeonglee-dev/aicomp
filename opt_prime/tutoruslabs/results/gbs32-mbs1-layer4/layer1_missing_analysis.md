# Layer 1 누락 원인 분석

## 로그 파일
- `results/20260115184943.log`
- 설정: 4 layers, PP=2, GBS=32, MBS=1

---

## 현상

COMBINED PROFILE 결과에서 layer_1이 누락됨:

```
[Per-Layer Breakdown]
Layer           Median(ms)   Min(ms)      Max(ms)      Mean(ms)     %
---------------------------------------------------------------------------
layer_0         1.3129       1.2873       1.7825       1.3771       15.67
layer_2         1.5970       1.5528       5.6148       2.0253       19.07
layer_3         1.3589       1.2834       1.7790       1.4473       16.22
```

**layer_1이 없음!**

---

## PP Stage 분할 상태

```
4 layers, PP=2 구성:

┌─────────────────────────────────────┐
│ Stage 0 (rank 0)                    │
│ - embed_tokens                      │
│ - layer_0 (q_proj_0 ~ down_proj_0)  │
│ - layer_1 (q_proj_1 ~ down_proj_1)  │  ← 마지막 레이어
└─────────────────────────────────────┘
                 ↓ PP 통신
┌─────────────────────────────────────┐
│ Stage 1 (rank 1)                    │
│ - layer_2 (q_proj_2 ~ down_proj_2)  │  ← 첫 번째 레이어
│ - layer_3 (q_proj_3 ~ down_proj_3)  │
│ - lm_head                           │
└─────────────────────────────────────┘
```

로그 확인:
```
[rank:0] last_node[stage#0] = model_layers_1_mlp_down_proj
[rank:0] last_node[stage#1] = lm_head
```

---

## 원인 분석

### LayerBlockProfiler 측정 방식

```
Layer N 시간 측정:
  시작: q_proj_N의 pre_hook
  종료: q_proj_(N+1)의 pre_hook (또는 lm_head의 pre_hook)
```

### 각 레이어별 측정 가능 여부

| Layer | 시작점 | 종료점 | 시작 Stage | 종료 Stage | 측정 가능 |
|-------|--------|--------|------------|------------|-----------|
| layer_0 | q_proj_0 | q_proj_1 | Stage 0 | Stage 0 | ✅ |
| **layer_1** | **q_proj_1** | **lm_head** | **Stage 0** | **Stage 1** | ❌ |
| layer_2 | q_proj_2 | q_proj_3 | Stage 1 | Stage 1 | ✅ |
| layer_3 | q_proj_3 | lm_head | Stage 1 | Stage 1 | ✅ |

### layer_1 누락 이유

```
Stage 0에서 layer_1 측정 시도:

  q_proj_1 (Stage 0)  ──────────→  lm_head (Stage 1)
       ↑                                ↑
   pre_hook 등록됨                 pre_hook 등록 안됨!
                                  (다른 프로세스에 있음)
```

1. **layer_1**은 Stage 0의 마지막 레이어
2. 시작점 `q_proj_1`은 Stage 0에 있음 → hook 등록 ✅
3. 종료점 `lm_head`는 Stage 1에 있음 → hook 등록 ❌
4. **Hook은 같은 프로세스 내에서만 작동**
5. PP stage 경계를 넘는 측정 불가능

### 로그에서 확인

```
[LayerBlockProfiler] layer_0: q_proj start → next_q_proj start
[LayerBlockProfiler] layer_1: q_proj start → lm_head start  ← lm_head가 다른 stage!
[LayerBlockProfiler] Registered layers: [0, 1]  ← 등록은 되었지만...
```

Stage 0 결과에서 layer_0만 출력됨:
```
[Per-Layer Breakdown (q_proj → next_q_proj)]
layer_0         1.3771       1.2873       1.7825       0.1500       103.45
```

---

## 문제 요약

```
PP Stage 경계에서 마지막 레이어 측정 불가:

Stage 0: [layer_0] [layer_1] ─────┐
                        ↑        │
                   측정 불가      │ PP 경계
                                 │
Stage 1:           ┌─────────────┘
                   ↓
         [layer_2] [layer_3] [lm_head]
```

- Stage 0의 마지막 레이어 (layer_1): 종료 마커가 Stage 1에 있어 측정 불가
- Stage 1의 첫 번째 레이어 (layer_2): 시작 마커가 Stage 1에 있어 측정 가능

---

## 해결 방안

### 1. PP=1로 프로파일링 (권장)

```bash
# 모든 레이어가 한 stage에 있으므로 전체 측정 가능
./run_docker.sh 1B 0 127.0.0.1 1 1 True 1 1 1 profile
```

### 2. Stage 경계 레이어 별도 처리

코드 수정 필요:
- Stage 0의 마지막 레이어: `down_proj`의 post_hook을 종료 마커로 사용
- 다음 stage의 q_proj 대신 현재 stage 내 마커 활용

### 3. Cross-stage 타이밍 동기화

- PP 통신을 통해 타이밍 정보 교환
- 구현 복잡도 높음

### 4. 레이어 수 조정

```
PP=2일 때 균등 분할 예시:
- 4 layers: Stage 0 (0,1), Stage 1 (2,3) → layer_1 누락
- 8 layers: Stage 0 (0-3), Stage 1 (4-7) → layer_3 누락

→ 누락되는 레이어는 항상 "Stage 0의 마지막 레이어"
```

---

## 결론

| 항목 | 설명 |
|------|------|
| **원인** | PP stage 경계를 넘는 hook 측정 불가 |
| **누락 레이어** | 각 stage의 마지막 레이어 (Stage 0의 layer_1) |
| **권장 해결책** | PP=1로 프로파일링 후 결과 활용 |
| **대안** | Stage 내 마커(down_proj)를 종료점으로 사용하도록 코드 수정 |
