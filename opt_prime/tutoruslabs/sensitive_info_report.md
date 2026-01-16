# 민감 정보 점검 리포트

## 점검 일시
- 2026-01-15

## 점검 대상
- `/home/ieg95/workspace/aicomp/opt_prime/tutoruslabs/docs/`
- `/home/ieg95/workspace/aicomp/opt_prime/tutoruslabs/results/`

---

## 점검 항목

| 항목 | 점검 패턴 | 결과 |
|------|----------|------|
| IP 주소 (localhost 제외) | `([0-9]{1,3}\.){3}[0-9]{1,3}` | 없음 |
| 비밀번호/토큰 | `password`, `secret`, `token`, `api_key`, `credential` | 없음 |
| HuggingFace 토큰 | `hf_*`, `huggingface.*token` | 없음 |
| 이메일 주소 | `@`, `.com`, `.org`, `.net` | 없음 |
| 사용자명/홈 디렉토리 | `/home/[a-zA-Z]+` | 없음 |
| **호스트명** | 서버명 패턴 | **발견** |

---

## 발견된 민감 정보

### 1. 호스트명 `s8`

#### 위치
`*_gpustats.log` 파일들 (gpustat 명령어 출력)

#### 영향 받는 파일
| 파일 경로 | 발견 건수 |
|----------|----------|
| `results/gbs2-mbs1-layer16/20260115183413_gpustats.log` | 다수 |
| `results/gbs32-mbs1-layer16/20260115183537_gpustats.log` | 다수 |
| `results/gbs32-mbs1-layer4/20260115185804_gpustats.log` | 다수 |
| **총계** | **79건** |

#### 예시
```
s8             Thu Jan 15 09:34:14 2026  560.35.05
[0] NVIDIA A100-SXM4-80GB | 45°C,   0 % |     0 / 81920 MB |
[1] NVIDIA A100-SXM4-80GB | 46°C,   0 % |     0 / 81920 MB |
```

#### 위험도
- **낮음**: 내부 서버명 노출
- 외부 공개 시 서버 식별 가능

---

## 안전 확인된 항목

### docs/ 디렉토리
| 파일 | 상태 |
|------|------|
| `LayerBlockProfiler_review.md` | 안전 |
| `FXLayerProfiler_review.md` | 안전 |
| `pow_layer_boundary_analysis.md` | 안전 |
| 기타 `.md` 파일들 | 안전 |

### results/ 디렉토리
| 파일 유형 | 상태 | 비고 |
|----------|------|------|
| `*.log` (메인 로그) | 안전 | IP는 127.0.0.1만 포함 |
| `*_memstats.log` | 안전 | 호스트명 없음 |
| `*_gpustats.log` | **주의** | 호스트명 `s8` 포함 |
| `*.json` | 안전 | 프로파일 결과만 포함 |
| `*.csv` | 안전 | 수치 데이터만 포함 |

---

## 권장 조치

### 옵션 1: 호스트명 마스킹
```bash
# s8 → hostname 으로 치환
sed -i 's/^s8 /hostname /g' results/*/*_gpustats.log
```

### 옵션 2: gpustats 파일 삭제
```bash
# gpustats 로그 파일 삭제
rm results/*/*_gpustats.log
```

### 옵션 3: .gitignore 추가
```bash
# .gitignore에 추가
echo "*_gpustats.log" >> .gitignore
```

---

## 결론

| 구분 | 상태 |
|------|------|
| docs/ | ✅ 안전 |
| results/ (메인 로그) | ✅ 안전 |
| results/ (gpustats) | ⚠️ 호스트명 노출 |

**조치 필요**: `*_gpustats.log` 파일의 호스트명 `s8` 처리 필요
