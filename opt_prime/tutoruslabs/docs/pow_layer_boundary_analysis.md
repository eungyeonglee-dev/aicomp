# pow 기반 레이어 경계 탐지 상세 분석

## 분석 개요
`llama1b_fx_graph.csv` (Llama 1B 모델의 FX Graph 전체 노드)를 기반으로 `FXLayerProfiler`의 pow 기반 레이어 경계 탐지 로직을 검증함.

---

## 1. 실제 FX Graph 노드 분포

### pow 노드 현황 (총 33개)
| pow 노드 | CSV Line | 용도 | 해당 레이어 |
|---------|----------|------|------------|
| pow_1 | 39 | layer_0 input_layernorm | layer_0 |
| pow_2 | 118 | layer_0 post_attention_layernorm | layer_0 |
| pow_3 | 134 | layer_1 input_layernorm | layer_1 |
| pow_4 | 213 | layer_1 post_attention_layernorm | layer_1 |
| ... | ... | ... | ... |
| pow_31 | 1464 | layer_15 input_layernorm | layer_15 |
| pow_32 | 1543 | layer_15 post_attention_layernorm | layer_15 |
| **pow_33** | **1559** | **model.norm (최종 RMSNorm)** | **lm_head** |

### q_proj 노드 현황 (총 16개, 레이어당 1개)
| q_proj 노드 | CSV Line | 레이어 |
|------------|----------|--------|
| model_layers_0_self_attn_q_proj | 52 | layer_0 |
| model_layers_1_self_attn_q_proj | 147 | layer_1 |
| model_layers_2_self_attn_q_proj | 242 | layer_2 |
| ... | ... | ... |
| model_layers_15_self_attn_q_proj | 1477 | layer_15 |

### model_norm & lm_head
| 노드 | CSV Line |
|------|----------|
| model_norm_weight (get_attr) | 1564 |
| lm_head (call_module) | 1569 |

---

## 2. pow 기반 레이어 탐지 공식 검증

### 현재 코드 로직
```python
if pow_count > 1:
    new_layer = (pow_count - 1) // 2
    if new_layer != current_layer and new_layer < len(layers):
        current_layer = new_layer
        current_component = f'layer_{current_layer}'
```

### 공식 검증: `new_layer = (pow_count - 1) // 2`

| pow_count | 계산 결과 | 실제 레이어 | 정확성 |
|-----------|----------|------------|--------|
| 1 | 조건 미충족 (pow_count > 1 false) | layer_0 시작 | ⚠️ |
| 2 | (2-1)//2 = 0 | layer_0 | ✅ |
| 3 | (3-1)//2 = 1 | layer_1 | ✅ |
| 4 | (4-1)//2 = 1 | layer_1 | ✅ |
| 5 | (5-1)//2 = 2 | layer_2 | ✅ |
| ... | ... | ... | ✅ |
| 31 | (31-1)//2 = 15 | layer_15 | ✅ |
| 32 | (32-1)//2 = 15 | layer_15 | ✅ |
| **33** | **(33-1)//2 = 16** | **model_norm (lm_head의 일부)** | **❌** |

---

## 3. 발견된 문제점

### 문제점 A: pow_33 (model.norm) 할당 오류 - 심각한 버그

**상황**: pow_33은 `model.norm` (최종 RMSNorm)의 pow 연산으로, lm_head 컴포넌트에 속해야 함.

**코드 구조 (L435-440)**:
```python
new_layer = (pow_count - 1) // 2
if new_layer >= len(self.module.model.model.layers):   # L436
    current_component = 'lm_head'                       # L437
if new_layer != current_layer:                          # L438 ← 별도의 if!
    current_layer = new_layer                           # L439
    current_component = f'layer_{current_layer}'        # L440 ← 덮어씀!
```

**실제 동작** (pow_33 처리 시):
1. `new_layer = (33-1)//2 = 16`
2. L436: `16 >= 16` → **True** → `current_component = 'lm_head'`
3. L438: `16 != 15` → **True** → `current_layer = 16`
4. L440: `current_component = 'layer_16'` ← **'lm_head' 덮어씀!**

**버그 원인**: 두 `if`문이 독립적이라 L437의 `'lm_head'` 할당이 L440에서 `'layer_16'`으로 덮어쓰임

**영향받는 노드** (Lines 1559-1568):
```
pow_33 (line 1559)        → 현재: layer_16 (존재하지 않음!) / 실제: lm_head
mean_32 (line 1560)       → 현재: layer_16 / 실제: lm_head
model_norm_weight (1564)  → 현재: layer_16 / 실제: lm_head
mul_179 (line 1567)       → 현재: layer_16 / 실제: lm_head
```

**결과**:
- `layer_16`이라는 존재하지 않는 컴포넌트가 생성됨
- 해당 노드들의 시간이 실제 컴포넌트 통계에서 누락됨
- lm_head의 시간이 심각하게 과소 측정됨

### 문제점 B: pow_1 경계 처리

**상황**: pow_1 발생 시 `pow_count > 1` 조건이 false이므로 레이어 전환이 일어나지 않음.

**현재 동작**:
1. pow_1: pow_count=1, 조건 불충족
2. pow_1 이후 ~ pow_2 이전 노드들은 이전 컴포넌트(embedding)에 할당됨
3. layer_0의 input_layernorm 관련 연산이 embedding으로 잘못 분류될 수 있음

---

## 4. 노드 순서 상세 분석

### Embedding → Layer_0 전환 구간
```
Line 35-51: embedding 관련 노드들
Line 39: pow_1 (layer_0 input_layernorm) ← pow_count=1, 조건 불충족
Line 40-51: 여전히 embedding으로 분류될 가능성
Line 52: model_layers_0_self_attn_q_proj ← layer_0 시작점으로 사용 가능
```

### Layer_15 → lm_head 전환 구간
```
Line 1543: pow_32 (layer_15 post_attention_layernorm)
Line 1544-1558: layer_15 FFN 관련 노드들
Line 1559: pow_33 (model.norm) ← new_layer=16, 조건 불충족으로 layer_15 유지
Line 1560-1568: model_norm 관련 노드들 (잘못된 할당)
Line 1569: lm_head ← 여기서야 lm_head로 전환
```

---

## 5. 권장 개선 방안

### 방안 1: if → elif 수정 (최소 수정)
```python
# L436-440 수정
if new_layer >= len(self.module.model.model.layers):
    current_component = 'lm_head'
elif new_layer != current_layer:   # if → elif 로 변경!
    current_layer = new_layer
    current_component = f'layer_{current_layer}'
```

**효과**: `new_layer >= num_layers`일 때 `'lm_head'`로 설정하고, 이후 조건을 건너뜀

### 방안 2: q_proj 기반 레이어 경계 탐지 (권장)
```python
# q_proj를 레이어 시작점으로 사용
if 'self_attn_q_proj' in node.name:
    match = re.search(r'layers_(\d+)', node.name)
    if match:
        current_layer = int(match.group(1))
        current_component = f'layer_{current_layer}'
```

**장점**:
- transformers 버전에 관계없이 안정적
- 레이어 번호가 노드 이름에 명시적으로 포함됨
- RMSNorm 구현 방식 변경에 영향받지 않음

### 방안 3: 하이브리드 접근법
```python
# Primary: q_proj 기반
if 'self_attn_q_proj' in node.name:
    # ... q_proj 기반 레이어 탐지
# Fallback: pow 기반 (q_proj 없는 경우)
elif node.target == 'pow' and not found_q_proj:
    # ... pow 기반 탐지
```

---

## 6. 결론

### 검증 결과 요약

| 항목 | 결과 |
|------|------|
| pow 기반 공식 (일반 케이스) | ✅ 정상 동작 (layer 0~15) |
| pow_1 경계 처리 | ⚠️ 잠재적 문제 (embedding/layer_0 경계) |
| pow_33 처리 (model.norm) | ❌ **심각한 버그** (`layer_16` 생성) |
| lm_head 시간 측정 정확도 | ❌ 심각하게 과소 측정 |

### 버그 핵심 원인

L436-440에서 두 `if`문이 독립적이라 `'lm_head'` 할당이 `'layer_16'`으로 덮어써짐:
```python
if new_layer >= len(layers):      # True → 'lm_head'
    current_component = 'lm_head'
if new_layer != current_layer:    # True → 'layer_16'으로 덮어씀!
    current_component = f'layer_{current_layer}'
```

### 최종 권고

1. **즉시 수정 필요**: L438의 `if` → `elif`로 변경
2. **장기적 개선**: q_proj 기반 탐지로 전환하여 안정성 확보
3. **Llama 1B에서의 영향**: model.norm 관련 4-5개 노드가 존재하지 않는 `layer_16`에 할당되어 통계에서 누락됨

---

## 부록: Llama 아키텍처에서의 pow 연산

Llama 모델에서 pow 연산은 RMSNorm에서만 발생:

```python
# RMSNorm 구현
def forward(self, x):
    variance = x.pow(2).mean(-1, keepdim=True)  # pow 연산 발생
    x = x * torch.rsqrt(variance + self.eps)
    return self.weight * x
```

각 Transformer Layer에는 2개의 RMSNorm이 있음:
1. `input_layernorm`: Self-Attention 전
2. `post_attention_layernorm`: FFN 전

따라서: `pow 개수 = (레이어 수 × 2) + 1 (model.norm)`
- Llama 1B (16 layers): 16 × 2 + 1 = 33개 pow ✅ 검증됨
