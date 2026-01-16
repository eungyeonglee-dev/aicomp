# FXLayerProfiler 클래스 코드 리뷰

## ⚠️ 상태: 제거됨

**FXLayerProfiler 클래스는 opt_prime/IR.py의 LayerProfileInterpreter와 중복되어 제거되었습니다.**

FX 노드 레벨 프로파일링이 필요한 경우:
```python
from opt_prime.IR import LayerProfileInterpreter
profiler = LayerProfileInterpreter(submod, use_cuda=True)
output = profiler.run(input_tensor)
profiler.print_layer_profile()
```

---

## 개요 (아카이브)
`FXLayerProfiler`는 FX Interpreter를 사용하여 노드 단위로 시간을 측정하고, 이를 embedding/layer_N/lm_head 컴포넌트로 그룹화하는 프로파일러였습니다.

---

## 장점

1. **정확한 노드 레벨 측정**: FX Interpreter의 `run_node`를 오버라이드하여 각 노드별 시간을 개별 측정
2. **통계적 견고성**: mean/median/min/max/std 모두 계산하여 outlier 영향 최소화
3. **micro-batch 정규화**: `num_mb`로 나누어 per-forward-pass 평균 제공

---

## 문제점 및 개선 제안

### 1. `_analyze_graph()` - pow 기반 레이어 경계 탐지 문제 (L378-458)

```python
elif pow_count > 1:
    new_layer = (pow_count - 1) // 2
```

**문제**: RMSNorm의 `pow` 연산을 레이어 경계로 사용하는데, 이 로직이 모든 Llama 버전에서 동일하게 동작한다고 보장할 수 없습니다. 특히:
- transformers 버전에 따라 FX graph 노드 이름/순서가 달라질 수 있음
- `pow` 노드가 RMSNorm 외의 연산에서도 발생할 수 있음

**제안**: `q_proj` 노드 기반 경계 탐지를 primary로, `pow` 기반을 fallback으로 사용

---

### 2. ~~`run_with_timing()` - 동기화 오버헤드~~ (L460-512) ✅ 적절한 구현

```python
end_event.record()
end_event.synchronize()  # 이 노드의 연산만 대기
```

**재검토 결과**: 이 구현은 **적절합니다**.

| 메서드 | 동작 | 오버헤드 |
|--------|------|---------|
| `torch.cuda.synchronize()` | 모든 스트림의 모든 연산 대기 | 높음 |
| `event.synchronize()` | 해당 이벤트 시점까지만 대기 | 낮음 |

현재 코드는 `event.synchronize()`를 사용하여:
- 해당 노드의 연산 완료까지만 대기
- 전역 동기화 대비 오버헤드 최소화
- 프로파일링 목적으로 적절한 trade-off

**경미한 오버헤드**: 노드당 CPU-GPU 이벤트 동기화 비용이 있으나, 프로파일링 정확도를 위해 감수할 만한 수준임

---

### 3. `node_to_component` 매핑 불완전 (L406-458)

```python
if node.op in ['placeholder', 'output']:
    continue
```

**문제**: `call_function` 노드 (예: `add`, `mul` for residual connection)가 어떤 컴포넌트에도 속하지 않을 수 있음

```python
if current_component:
    self.node_to_component[node_name] = current_component
```

`current_component`가 `None`이면 노드가 무시됨 → 시간 측정 누락

---

### 4. 첫 노드 동기화 위치 (L483-485)

```python
if inner_self.first_node:
    torch.cuda.synchronize()
    inner_self.first_node = False
```

**문제**: `placeholder`, `output` 노드는 스킵되므로 첫 번째 실제 연산 노드에서 동기화됨. 하지만 이미 이전 micro-batch의 연산이 GPU에서 실행 중일 수 있어 측정이 부정확해질 수 있음

---

### 5. 통계 계산 - std 계산 (L536)

```python
'std_ms': (sum((t - mean_val)**2 for t in per_fwd_times) / len(per_fwd_times)) ** 0.5 if len(per_fwd_times) > 1 else 0,
```

**문제**: sample std (n-1로 나눔)가 아닌 population std (n으로 나눔) 사용. 측정 횟수가 적으면 과소 추정됨

---

### 6. `print_node_breakdown()` 미사용 (L625-648)

이 메서드는 정의되어 있지만 main 코드에서 호출되지 않음. 유용한 디버깅 정보를 제공하므로 활용 검토 필요

---

## 비교: LayerBlockProfiler vs FXLayerProfiler

| 항목 | FXLayerProfiler | LayerBlockProfiler |
|------|-----------------|-------------------|
| 측정 방식 | 모든 FX 노드 개별 측정 | 경계 hook 기반 (start/end) |
| 정확도 | 노드별 상세 | 전체 레이어 시간 |
| 오버헤드 | 중간 (event sync per node) | 낮음 |
| 용도 | 노드 분석/디버깅 | 실제 성능 측정 |

---

## 결론

`FXLayerProfiler`는 FX graph 노드 단위 분석에 유용합니다.

**주요 문제점**:
- **pow 기반 레이어 경계 탐지**: pow_33(model.norm)이 layer_15에 잘못 할당되는 문제 (상세 분석: `pow_layer_boundary_analysis.md`)
- **node_to_component 매핑 불완전**: 일부 노드가 누락될 수 있음

**적절한 구현**:
- ~~동기화 오버헤드~~: `event.synchronize()` 사용으로 최소화됨

실제 레이어 시간 측정에는 `LayerBlockProfiler`가 더 적합하고, `FXLayerProfiler`는 디버깅/노드별 상세 분석 용도로 활용하는 것이 좋습니다.
