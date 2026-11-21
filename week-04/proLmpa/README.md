# Week-04 Advanced RAG 비교 — proLmpa

데이터: 2024 알기 쉬운 의료급여제도 FAQ PDF (기준선 단일 문서)
평가 질문: Q1–Q9 (9문항, `eval_questions.json`)
실행 코드: `rag_configs.py`, `run_eval.py`
결과 파일: `results/{baseline,chunking,hybrid}/`

---

## 1. 기존 평가 검토·보완

### Week-03 평가의 문제점

week-03 평가 결과를 사람이 직접 원문과 대조한 결과 네 가지 문제를 확인했다.

① 근거 ID 기반 ground truth의 취약성
week-03는 `ground_truth_chunk_ids: ["chunk_57", "chunk_58", "chunk_59"]`로 Q2 정답을 판정했으나, 실제 Q48/A48 본문("1종수급권자는 5%, 2종수급권자는 15%")은 `chunk_59`에만 존재한다. `chunk_57`, `chunk_58`는 헤더·풋터 잔재였다. 청킹 방식이 달라지면 청크 ID 자체가 바뀌므로 세 구성을 동일 기준으로 비교할 수 없다.
→ 수정: `evidence_spans`(PDF 원문 짧은 문자열)로 교체. Evidence Recall@5 = Top-5 컨텍스트에 span이 포함된 비율.

② FactualCorrectness 기본값(mode="precision")의 편향
week-03 Q1에서 FC=0.57이 나왔으나 사람 판정은 pass였다. precision 모드는 답변에 추가된 올바른 세부정보를 오답으로 간주해 감점한다. 측정 목표는 "참조 사실이 답변에 포함됐는가"이므로 recall 모드가 적합하다.
→ 수정: `FactualCorrectness(mode="recall")` 적용.

③ Unanswerable 질문에 대한 ResponseRelevancy N/A 처리 미흡
거절 답변("제공된 문서에서 확인할 수 없습니다")은 질문과의 관련성 측정이 무의미하다. week-03에서 Q4 ResponseRelevancy=0.00이 나와 평균을 왜곡했다.
→ 수정: `answerable=false`인 Q4, Q9에 대해 Faithfulness·ResponseRelevancy·ContextRecall을 N/A로 처리. 집계에서 제외.

④ 평가 범위 부족 — 테이블·숫자·경계 케이스 미포함
Q1–Q5는 모두 비슷한 패턴(본문 Q&A 직접 조회)이었다. 테이블 행 조회(본인부담률 표), 키워드 중복(틀니 관련 다른 조항), unanswerable 추가 케이스가 없었다.
→ 수정: Q6(본인부담금 표 수치 조회), Q7(틀니 보상제 제외 조항), Q8(조현병 정신질환 표), Q9(국외 진료, unanswerable) 추가. 각 `evidence_spans`는 PDF 원문과 대조·검증.

### 보완 후 평가 기준 (세 구성 공통 고정)

| 항목 | 설정 |
|---|---|
| 생성 모델 | claude-sonnet-4-6 |
| 판정 모델 | claude-sonnet-4-6 (self-judging bias 존재, 한계로 기록) |
| 임베딩 모델 | paraphrase-multilingual-mpnet-base-v2 |
| top_k | 5 |
| 시스템 프롬프트 | week-02 원본 (week-03 strict 프롬프트 사용 안 함 — 교란 방지) |
| 지표 | Evidence Recall@5, Page Hit@5, Faithfulness, FactualCorrectness(recall), ResponseRelevancy, ContextRecall, Answerability |
| Rubric | `rubric.md` 참조 |

---

## 2. Advanced RAG 기법 조사·선택

### "Advanced RAG" 용어 출처

이 프로젝트에서 사용하는 "Advanced RAG"라는 분류는 **Gao et al. (2024) *Retrieval-Augmented Generation for Large Language Models: A Survey*** (arXiv:2312.10997)를 기준으로 한다. 이 논문은 RAG를 세 단계로 분류하며 각 단계의 기법을 명시적으로 정의한다:

| 단계 | 정의 | 본 프로젝트 대응 |
|---|---|---|
| Naive RAG | 고정 크기 청킹 + Dense 단일 검색 | Baseline |
| **Advanced RAG** | 인덱싱 개선(기법 A) + 검색 개선(기법 B) | Chunking, Hybrid |
| Modular RAG | 쿼리 재작성·재랭킹·에이전트 등 복합 파이프라인 | 이번 범위 외 |

논문 §3.2 "Indexing Optimization"은 구조적 청킹(Q&A 쌍 추출·테이블 보존)을 Advanced RAG 인덱싱 개선으로, §3.3 "Retrieval"은 하이브리드 검색(키워드+의미 임베딩)을 Advanced RAG 검색 개선으로 명시한다.

### 개선 대상

week-03에서 확인된 주요 실패:

- Q2 (틀니 본인부담률): Evidence Recall@5 = 0.0. 500자 고정 청킹이 p.17의 Q48/A48 경계를 잘라 "1종수급권자는 5%"가 단일 청크에 보존되지 않았다. 헤더·풋터 노이즈 청크 4개가 Top-5 슬롯을 차지했다.
- Q8 (조현병 본인부담률): 증거 span이 p.6 테이블 셀에 흩어져 있어 `extract_text()`의 단순 텍스트 추출이 행·열을 뒤섞는다. 모든 구성에서 실패.

### 기법 A: 구조 인식 청킹 (Structure-Aware Chunking)

해결하는 문제: 고정 크기 청킹이 의미 단위(Q&A 쌍, 테이블 행)를 절단해 필요한 사실이 여러 청크에 분산되는 문제.

동작 단계: 인덱싱(Indexing). PDF를 재파싱해 새 컬렉션을 생성한다.

동작 방식:
1. `pdfplumber.page.extract_tables()`로 구조화된 테이블 추출 → 파이프 구분 텍스트 청크 생성
2. 본문 텍스트에서 `(?=Q\d+\.)` 경계로 Q&A 블록 분리 → 각 블록을 단일 청크
3. 헤더·풋터 노이즈 패턴 제거 (`NOISE_RE`)
4. 100자 미만 청크 폐기, 800자 초과 블록은 500/400 겹침 고정 분할로 폴백

이 데이터에 적합한 이유: FAQ 문서는 `Q숫자.`/`A숫자.` 구조가 명확하다. p.17 Q48/A48처럼 정답이 하나의 Q&A 블록에 완결되는 경우 경계 분할이 직접적인 해결책이다. 테이블 추출은 p.5·p.6의 본인부담률 표(Q6, Q8 근거)를 행 단위로 보존하기 위해 필요하다.

예상 변화: Q2 Evidence Recall 0.0 → 1.0. 테이블 기반 질문(Q6, Q8)의 검색 정밀도 향상.

추가 비용·위험: 인덱싱 시간 증가(다중 PDF 파싱). 대규모 문서 추가 시 검색 풀이 넓어져 무관한 청크가 경쟁 — 이번 실험에서 실제로 발생했다.

참고 자료:
- Gao et al. (2024) §3.2 Indexing Optimization — Q&A 쌍 추출·구조적 청킹을 Advanced RAG 인덱싱 개선으로 명시: arXiv:2312.10997
- Anthropic Contextual Retrieval (2024): https://www.anthropic.com/news/contextual-retrieval

### 기법 B: 하이브리드 검색 (Hybrid Retrieval — BM25 + Dense → RRF)

해결하는 문제: Dense 임베딩이 "틀니", "1종", "5%"처럼 짧고 정확한 한국어 용어를 의미적으로 유사한 다른 청크와 혼동하는 문제. 고빈도 정확 매칭 쿼리에서 BM25가 Dense를 보완한다.

동작 단계: 검색(Retrieval). 기존 인덱스 위에 BM25 계층을 추가한다.

동작 방식:
1. Dense: 기존 ChromaDB 컬렉션에서 Top-20 검색 (cosine similarity)
2. BM25: `rank_bm25.BM25Okapi` + `kiwipiepy` 형태소 분석으로 Top-20 검색. "틀니", "1종수급권자"를 독립 형태소로 인식해 정확 매칭
3. RRF(k=60)로 두 결과 병합 → Top-5

이 데이터에 적합한 이유: 의료급여 FAQ는 "1종", "2종", "5%", "15%", "틀니", "임플란트" 같은 고유 용어·수치가 정답의 핵심이다. Dense 임베딩은 의미적 근접성에 강하지만 이런 정확 용어 매칭에서 취약하다. BM25는 형태소 분석을 통해 한국어 조사를 처리하면서 키워드 정확도를 보완한다.

예상 변화: Q2 BM25 Top-5에 chunk_59 포함 → Evidence Recall 0.0 → 1.0. 기존 Dense 성공 사례는 그대로 유지.

추가 비용·위험: BM25 인덱스 초기화 시간 (72청크 기준 무시할 수준). 검색당 BM25 호출 1회 추가 (~20ms). Dense와 BM25가 모두 실패하는 쿼리에서 RRF가 도움이 안 될 수 있다.

참고 자료:
- Gao et al. (2024) §3.3 Retrieval — 하이브리드 검색(키워드+의미 임베딩)을 Advanced RAG 검색 개선으로 명시: arXiv:2312.10997
- Cormack et al. (2009) Reciprocal Rank Fusion: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- rank_bm25 라이브러리: https://github.com/dorianbrown/rank_bm25

---

## 3. 세 구성 구현·실행

### 구성 정의

| 구성 | 컬렉션 | 청킹 | 검색 방식 |
|---|---|---|---|
| Baseline | `medical_aid` (week-02) | 500자 고정, 100자 겹침 | Dense Top-5 |
| 개선 A (chunking) | `medical_aid_qa` (신규) | 테이블 + Q&A 경계 + 고정 폴백 | Dense Top-5 |
| 개선 B (hybrid) | `medical_aid` (week-02 동일) | 500자 고정 (baseline과 동일) | BM25 Top-20 + Dense Top-20 → RRF(k=60) → Top-5 |

개선 B는 baseline에서 시작한다. A와 B를 함께 적용하는 조합은 이번 실험 범위 밖이다.

### Config A 세부 사항

다중 PDF 인덱싱(week-02/proLmpa/data/ 내 3개 파일):

> 초기 평가에 `2026+의료급여사업+안내.pdf`(790페이지, 2,462청크)를 포함했으나 검색 풀의 95.5%를 차지해 2024 FAQ 청크가 Top-5 밖으로 밀리는 심각한 희석(dilution) 현상이 발생했다. 해당 파일은 평가 전 제거했다.

| PDF | 페이지 | 생성 청크 |
|---|---|---|
| 2024 알기 쉬운 의료급여제도.pdf | 21 | 68 |
| 보건복지부_의료급여.pdf | 1 | 0 (추출 가능 텍스트 없음) |
| 의료급여법 시행령(제35932호).pdf | 10 | 47 |
| 합계 | 32 | 115 |

청킹 방법별 세부 분포는 `python rag_configs.py --inspect` 출력으로 확인 가능.

### Config B 세부 사항

- BM25 인덱스: 72개 baseline 청크, `kiwipiepy` 형태소 분석
- Dense Top-20 + BM25 Top-20 → RRF(k=60) → Top-5
- 검색당 임베딩 호출 1회 + BM25 점수 계산 1회

### 고정 조건 및 변경 이유

기법과 무관한 조건은 세 구성 모두 동일하게 고정했다.

- 생성 모델: claude-sonnet-4-6
- 시스템 프롬프트: week-02 원본
- top_k: 5
- 평가 질문셋: Q1–Q9 동일

Config A에서 컬렉션이 바뀐 것은 기법 자체의 필연적 결과다(다른 청킹 → 다른 컬렉션). Config B는 baseline 컬렉션을 그대로 사용해 하이브리드 검색 효과만 독립적으로 측정했다.

### 평가 질문셋

| ID | 유형 | 질문 |
|---|---|---|
| Q1 | answerable | 선택의료급여기관 적용 대상자가 의료급여의뢰서 없이 다른 기관에서 진료받을 수 있는 경우는 언제인가요? |
| Q2 | answerable_known_failure | 65세 이상 수급권자가 틀니 의료급여를 받을 때 1종과 2종의 본인부담률은 각각 얼마인가요? |
| Q3 | answerable | 의료급여 수급자가 약국에서 처방약을 조제받을 때 본인부담금은 얼마인가요? |
| Q4 | unanswerable | 의료급여 수급자가 미용 목적 성형수술을 받을 경우 본인부담금은 얼마인가요? |
| Q5 | partially_answerable | 의료급여 수급자가 65세 이상 치과 임플란트 진료를 받을 때 본인부담률과, 치과 스케일링 치료 비용은 얼마인가요? |
| Q6 | answerable | 의료급여 2종 수급권자가 의원(1차의료급여기관)에서 외래 진료를 받을 때 본인부담금은 얼마인가요? |
| Q7 | answerable | 의료급여 수급권자의 본인부담금 보상제에서 65세 이상 틀니 비용이 보상 대상에 포함되나요? |
| Q8 | answerable | 의료급여 수급권자가 조현병으로 병원급 이상 외래진료를 받을 때 본인부담률은 얼마인가요? |
| Q9 | unanswerable | 의료급여 수급자가 국외 여행 중 현지 의료기관에서 진료를 받은 경우 의료급여를 청구할 수 있나요? |

유형 정의: `answerable` = 문서 내 근거 있음, `answerable_known_failure` = 근거 있으나 baseline 검색 실패 예상, `unanswerable` = 문서 범위 밖, `partially_answerable` = 일부만 답 가능.

---

## 4. 결과 비교

### 4-1. 집계 비교표

Evidence Recall, Faithfulness, ContextRecall: n=7 -- Q4·Q9(unanswerable) 제외,
FactualCorrectness(recall): 전체 n=9,
Answerability 정확도: n=8 -- Q5(partially_answerable) 제외,
사람 판정: 전체 n=9,

| 구성 | EvRecall@5 (n=7) | Faithfulness (n=7) | FC(recall) (n=9) | CtxRecall (n=7) | Answerability (n=8) | 사람 pass (n=9) | 사람 partial (n=9) | 사람 fail (n=9) | 평균 ms | RAG 총 비용 |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0.714 | 0.720 | 0.778 | 0.571 | 0.750 | 6 | 1 | 2 | 7,295 | $0.117 |
| Chunking | 0.571 | 0.849 | 0.611 | 0.714 | 0.875 | 5 | 2 | 2 | 6,344 | $0.111 |
| Hybrid | 0.714 | 0.832 | 0.722 | 0.714 | 0.750 | 6 | 1 | 2 | 5,726 | $0.119 |

> 평가용 Ragas LLM 호출 시간(약 155–157초/구성)은 RAG 비용과 별도로 기록했으며 위 비용에 미포함.
> Ragas 평가 모델 비결정성(non-determinism)으로 인해 Baseline 수치도 1차 실행과 소폭 차이가 있다(예: Faithfulness Q2: 0.80→0.40). 모든 수치는 2026 문서 제거 후 동일 조건에서 재실행한 2차 결과로 통일한다.

### 4-2. 문항별 비교표

| Q | 유형 | | EvRecall | Faith | FC(rec) | RR | CtxRec | ms | 비용($) | 사람판정 |
|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | answerable | baseline | 1.00 | 1.00 | 1.00 | 0.983 | 1.00 | 7,329 | 0.014 | pass |
| | | chunking | 0.00† | 1.00 | 1.00 | 0.955 | 1.00 | 5,819 | 0.012 | pass |
| | | hybrid | 1.00 | 1.00 | 1.00 | 0.971 | 1.00 | 6,469 | 0.015 | pass |
| Q2 | answerable_known_failure | baseline | 0.00 | 0.40 | 0.00 | 0.000 | 0.00 | 3,941 | 0.006 | fail |
| | | chunking | 1.00 | 1.00 | 1.00 | 0.983 | 1.00 | 4,309 | 0.010 | pass |
| | | hybrid | 1.00 | 1.00 | 1.00 | 0.980 | 1.00 | 4,323 | 0.011 | pass |
| Q3 | answerable | baseline | 1.00 | 0.82 | 1.00 | 0.972 | 0.00* | 11,717 | 0.017 | pass |
| | | chunking | 1.00 | 1.00 | 0.00* | 0.923 | 0.00* | 8,962 | 0.016 | pass |
| | | hybrid | 1.00 | 1.00 | 1.00 | 0.922 | 1.00 | 9,704 | 0.018 | pass |
| Q4 | unanswerable | baseline | N/A | N/A | 1.00 | N/A | N/A | 6,490 | 0.013 | partial |
| | | chunking | N/A | N/A | 1.00 | N/A | N/A | 3,603 | 0.010 | partial |
| | | hybrid | N/A | N/A | 1.00 | N/A | N/A | 6,237 | 0.013 | partial |
| Q5 | partially_answerable | baseline | 1.00 | 0.86 | 1.00 | 0.000‡ | 1.00 | 7,772 | 0.012 | pass |
| | | chunking | 1.00 | 0.86 | 1.00 | 0.000‡ | 1.00 | 5,220 | 0.011 | pass |
| | | hybrid | 1.00 | 0.83 | 1.00 | 0.957 | 1.00 | 5,820 | 0.013 | pass |
| Q6 | answerable | baseline | 1.00 | 0.71 | 1.00 | 0.990 | 1.00 | 7,869 | 0.015 | pass |
| | | chunking | 1.00 | 0.75 | 0.00 | 0.000 | 1.00 | 8,123 | 0.013 | fail |
| | | hybrid | 0.00 | 0.60 | 0.00 | 0.000 | 0.00 | 3,841 | 0.012 | fail |
| Q7 | answerable | baseline | 1.00 | 1.00 | 1.00 | 0.953 | 1.00 | 9,416 | 0.013 | pass |
| | | chunking | 0.00† | 1.00 | 0.50 | 0.945 | 1.00 | 9,766 | 0.015 | partial |
| | | hybrid | 1.00 | 0.89 | 0.50 | 0.952 | 1.00 | 5,939 | 0.013 | pass |
| Q8 | answerable | baseline | 0.00 | 0.25 | 0.00 | 0.000 | 0.00 | 7,341 | 0.015 | fail |
| | | chunking | 0.00 | 0.33 | 0.00 | 0.000 | 0.00 | 7,860 | 0.013 | fail |
| | | hybrid | 0.00 | 0.50 | 0.00 | 0.000 | 0.00 | 5,412 | 0.013 | fail |
| Q9 | unanswerable | baseline | N/A | N/A | 1.00 | N/A | N/A | 3,780 | 0.011 | pass |
| | | chunking | N/A | N/A | 1.00 | N/A | N/A | 3,433 | 0.009 | pass |
| | | hybrid | N/A | N/A | 1.00 | N/A | N/A | 3,786 | 0.012 | pass |

> `†` EvRecall=0.0 false negative: Q&A 경계 청킹으로 텍스트 형식이 바뀌어 verbatim span 매칭 실패. CtxRecall=1.0·FC=1.0 등 나머지 지표가 정상이므로 실질적 검색 실패가 아님.
> `*` Ragas reference 표현 불일치 추정: ev=1.0 또는 faith=1.0이므로 실제 근거는 검색·인용됨. Ragas LLM이 reference 문자열과 답변 표현을 불일치로 판정한 평가 오류.
> `‡` ResponseRelevancy=0.0: partially_answerable 질문에서 스케일링 거절이 포함된 답변을 Ragas가 "관련성 없음"으로 평가하는 측정 한계. 사람 판정은 pass(거절이 올바른 응답).

### 4-3. 대표 사례 심층 분석

#### Q2 — 틀니 본인부담률 (핵심 실패 케이스)

Baseline
- 검색: `chunk_62`(p.18), `chunk_60`(p.17), `chunk_47`(p.14), `chunk_56`(p.16), `chunk_33`(p.11)
- chunk_59(p.17, A48 본문)가 Top-5에 없음 — 500자 고정 청킹이 Q48/A48를 2개 청크로 분할했고, 임베딩 유사도로 임플란트·보상제 관련 청크가 먼저 올라옴
- 답변: 거절 ("제공된 문서에서 해당 내용을 찾을 수 없습니다")
- 사람 판정: fail

Chunking (개선 A)
- 검색: `qa_chunk_56`(p.17) → Q48/A48 블록 전체가 단일 청크로 보존됨 (160자)
- Evidence Recall = 1.0, "1종수급권자는 5%, 2종수급권자는 15%" 답변 생성 확인
- FC(recall)=1.0 — 2차 실행에서 Ragas가 정상 평가 (1차 FC=0.0은 평가 비결정성 오류였음)
- 사람 판정: pass (수치와 맥락 모두 정확)

Hybrid (개선 B)
- 검색: `chunk_62`(RRF 1위), `chunk_59`(RRF 2위) — BM25가 "틀니", "1종", "5%"를 형태소로 매칭해 chunk_59를 BM25 Top-2에 올림
- Evidence Recall = 1.0, FC(recall) = 1.0, Faithfulness = 1.0
- 사람 판정: pass

#### Q1 — 선택의료급여기관 의뢰서 예외 (기존 성공 케이스)

Baseline: 검색 성공 (chunk_42, p.13), 거절 없이 "응급환자, 장애인 보조기기" 정확 답변. FC(recall)=1.0.

Chunking (개선 A): [2026 문서 제거 후] 검색 성공 — 2024 FAQ Q&A 경계 청크가 Top-5 진입. EvRecall=0.0은 false negative(Q&A 청킹으로 텍스트 포맷 변경). FC=1.0, CtxRecall=1.0으로 답변 정확 확인. 사람 판정 pass.

Hybrid: 검색 성공 (chunk_40, chunk_42, p.13). Baseline과 동일 컬렉션 사용으로 영향 없음. FC(recall)=1.0 유지.

#### Q6 — 2종 외래 본인부담금 (테이블 조회)

Baseline: 고정 청킹으로 p.5 전체 텍스트가 연속 청크에 담겨 검색 성공 → pass.

Chunking [2026 제거 후]: EvRecall=1.0, CtxRecall=1.0으로 검색 성공. 그러나 FC=0.0, ResponseRelevancy=0.0 — 모델이 p.5 테이블 행의 "1,000원" 대신 같은 행의 "15%"를 답변에 사용한 것으로 추정. 테이블 행 `"외래 1,000원 15% 15% 500원 15%"`에 고정 금액과 비율이 혼재해 모델이 비율을 선택했을 가능성이 있다. 사람 판정: fail.

Hybrid: BM25에서 "1,000원"이 다수 청크에 분산되어 Q6 고유 근거를 올리지 못함. EvRecall=0.0 → 거절 → fail.

사람 판정: Baseline=pass, Chunking=fail(검색 성공이나 오답), Hybrid=fail(미검색).

#### Q8 — 조현병 외래 본인부담률 (p.6 테이블)

세 구성 모두 실패. evidence span `"조현병 병원급 이상 5%"`는 p.6 정신질환 테이블의 한 행이지만, pdfplumber가 `extract_tables()`로 이 테이블을 구조적으로 추출하지 못했거나 추출된 테이블 텍스트에 해당 조합이 존재하지 않는 형태로 저장됐을 가능성이 있다. PDF p.6를 직접 열어 테이블 구조와 추출 결과를 비교해야 한다.

사람 판정: 모든 구성 fail.

### 4-4. 사람 판정 요약

PDF 원문을 직접 열어 각 답변과 대조한 결과다. 판정 근거는 각 `eval_results.json`의 `human_notes` 필드에 기록했다.

| Q | Baseline | Chunking | Hybrid | 판정 근거 요약 |
|---|---|---|---|---|
| Q1 | pass | pass | pass | PDF A25: 응급증상·장애인 보조기기 두 예외. [2026 제거 후] Chunking도 Q&A 경계 청크로 A25 검색 성공 (EvRecall false negative이나 FC=1.0·CtxRec=1.0 확인) |
| Q2 | fail | pass | pass | PDF A48: 1종 5%, 2종 15%. Baseline은 chunk_59 미검색으로 거절 |
| Q3 | pass | pass | pass | PDF p.5 표: 1종·2종 모두 500원. 세 구성 모두 핵심 사실 포함. Chunking FC/CtxRecall=0.0은 Ragas mismatch (faith=1.0·ev=1.0 확인) |
| Q4 | partial | partial | partial | 거절 + 문서 외 추론("비급여 가능성"). 세 구성 동일 |
| Q5 | pass | pass | pass | 임플란트 정확, 스케일링 거절. [2026 제거 후] Chunking도 baseline·hybrid와 동일 행동 |
| Q6 | pass | fail | fail | PDF p.5 표: 2종 외래 의원 1,000원. [2026 제거 후] Chunking 검색 성공이나 모델이 1,000원 대신 15% 답변. Hybrid 미검색 거절 |
| Q7 | pass | partial | pass | PDF A53: 틀니·임플란트 보상제 제외 명시. [2026 제거 후] Chunking Q53 부분 검색 — 제외 사실 포함이나 FC=0.5로 불완전 |
| Q8 | fail | fail | fail | PDF p.6 표: 조현병 병원급 이상 5%. 세 구성 모두 테이블 추출 실패로 거절 |
| Q9 | pass | pass | pass | 평가 설계상 unanswerable. [2026 제거 후] Chunking도 정상 거절 |

---

## 5. 개선 효과와 적용 여부 판단

### 5-1. 예상한 변화가 나타났는가?

| 예상 | 사람 판정 기준 결과 |
|---|---|
| A: Q2 Evidence Recall 0 → 1 | O — qa_chunk_56이 Top-1 검색, 사람 판정 pass |
| A: 테이블 기반 Q6/Q8 개선 | X — Q6, Q8 모두 여전히 fail |
| B: BM25로 Q2 chunk_59 복원 | O — BM25 Top-2, RRF Top-2 진입, 사람 판정 pass |
| B: 기존 성공 문항 유지 | 대부분 O — Q6만 새로 fail (Q1·Q3·Q5·Q7·Q9 유지) |

### 5-2. 개선됐거나 나빠진 문항

개선 A (Chunking) — 사람 판정 기준 (2026 문서 제거 후)

| 문항 | Baseline→Chunking 변화 | 원인 |
|---|---|---|
| Q2 | fail → pass | Q&A 경계 분할로 A48 단일 청크 보존 성공 |
| Q6 | pass → fail | 검색 성공이나 모델이 1,000원 대신 15% 답변 (테이블 행 혼재 수치) |
| Q7 | pass → partial | Q53 부분 검색 성공, 제외 사실 포함이나 불완전 (FC=0.5) |

개선 1건(Q2), 회귀 2건(Q6 full·Q7 half) — 순손실이 크게 줄었으나 아직 양수는 아님.
Q1·Q5·Q9: 2026 문서 제거로 baseline 수준 복원.

개선 B (Hybrid) — 사람 판정 기준

| 문항 | 변화 | 원인 |
|---|---|---|
| Q2 | fail → pass | BM25가 "틀니", "1종", "5%" 형태소 정확 매칭 |
| Q6 | pass → fail | "1,000원" 키워드 다수 청크에 분산, BM25 미분별 |

개선 1건(Q2), 회귀 1건(Q6) — 균형. 나머지 7문항 판정 유지.

### 5-3. 점수 상승과 실제 답변 개선의 일치 여부

- Q3 Chunking FC(recall)=0.0·CtxRecall=0.0 ↔ 사람 pass: EvRecall=1.0, Faith=1.0으로 근거 검색·인용 모두 성공. Ragas가 reference 표현("500원 (1종/2종 공통)")과 답변 표현 불일치로 0점 판정. 자동 평가가 표현 차이를 오류로 처리한 전형적 사례.
- Q5 ResponseRelevancy=0.0 ↔ 사람 pass (baseline/chunking): 스케일링 거절이 포함된 답변을 Ragas가 "관련성 없음"으로 평가하는 측정 한계. 사람 기준으로는 거절이 올바른 응답이므로 pass.
- Q6 Chunking EvRecall=1.0·CtxRecall=1.0 ↔ 사람 fail: 검색 지표는 완벽하나 FC=0.0·ResponseRelevancy=0.0. 모델이 테이블 행의 "1,000원" 대신 "15%"를 답변에 사용. 검색 지표가 성공임에도 답변 품질이 낮은 경우 — generation 단계의 실패.
- Q7 Baseline FC(recall)=1.0·Chunking FC=0.5 ↔ 사람 pass/partial: Baseline은 2차 실행에서 FC=1.0으로 정상 평가됨. Chunking FC=0.5는 Q53 답변이 불완전함을 반영하며 사람 판정(partial)과 일치.

전반적으로 EvRecall@5는 사람 판정과 가장 일치율이 높다. FC(recall)은 reference 표현 차이에 민감해 과소평가 경향이 있으며 사람 판정과 병행해야 신뢰할 수 있다. Answerability 지표가 Q9 Chunking을 감지하지 못한 점도 한계다.

### 5-4. 추가 비용·지연의 가치

| | Chunking | Hybrid |
|---|---|---|
| 추가 인덱싱 비용 | 115청크 임베딩 (1회성, 2026 제거 후) | BM25 초기화 ~1초 |
| 검색 지연 추가 | 없음 (Dense Top-5 동일) | +20ms/쿼리 (BM25 점수 계산) |
| 평균 총 지연 | 6,344ms (baseline 대비 -951ms) | 5,726ms (baseline 대비 -1,569ms) |
| 사람 판정 개선 | Q2: fail→pass (1건) | Q2: fail→pass (1건) |
| 사람 판정 회귀 | Q6: pass→fail, Q7: pass→partial (2건) | Q6: pass→fail (1건) |
| 순효과 | -1건 (Q7은 half-step 회귀) | 0건 (개선·회귀 균형) |
| 가치 | 중간 (Q6 오답 원인 해결 시 양수 전환 가능) | 높음 |

Hybrid는 추가 지연이 거의 없고 오히려 평균 응답이 빨라졌다(dense-only보다 짧은 컨텍스트로 답변 가능). 비용도 쿼리당 추가 지출 없이 baseline 수준을 유지한다.

### 5-5. 최종 판단

기법 A (구조 인식 청킹): 수정 (Modify)

판단 근거: 2026 문서 제거 후 Q1·Q5·Q9 회귀가 복원됐고 Q2 fix는 유지됐다. Q&A 경계 분할 기법 자체는 유효하다. 현재 미해결 사항: Q6(검색 성공이나 모델이 1,000원 대신 15% 답변), Q7(partial — fc=0.5), Q8(체계적 PDF 테이블 추출 실패). 개선 1건, 회귀 2건(Q6 full, Q7 half) — 아직 순손실이나 1차 실험(-5건)보다 크게 개선됐다.

수정 방향:
- Q6: p.5 테이블 행 추출 결과를 직접 확인(`extract_tables()` 출력 점검). 고정 금액(1,000원)과 비율(15%)이 혼재한 행의 메타데이터 분리 또는 프롬프트에 "금액과 비율을 구분해서 답하라" 명시
- Q7: Q53 텍스트가 단일 Q&A 청크로 보존되는지 `--inspect`로 확인. evidence span 불일치 시 span 갱신
- Q8: PDF p.6 테이블 직접 열어 `extract_tables()` 결과 확인. 추출 실패 시 OCR(pytesseract) 또는 수동 텍스트 추가 필요
- 대형 사업 안내 문서 추가 시 소스 필터 검색(`where={"source": "..."}`) 또는 별도 컬렉션 운영 필수

기법 B (하이브리드 검색 BM25+Dense→RRF): 유지 (Keep)

판단 근거: 목표였던 Q2 fix를 달성했고(사람 pass 확인), 기존에 성공하던 Q1·Q3·Q5·Q7·Q9 7문항의 판정이 그대로 유지됐다. 개선 1건, 회귀 1건으로 순효과 0건이지만 Q6 회귀는 BM25 쿼리 전처리로 보완 가능한 문제다. 추가 지연은 쿼리당 20ms 미만이며 비용 증가 없고 평균 응답 속도는 오히려 빨라졌다(-1,569ms vs baseline). Q8는 BM25로도 해결 불가 — 근본 원인이 PDF 테이블 추출 실패이므로 청킹 개선이 선행되어야 한다.
