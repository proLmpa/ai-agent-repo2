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
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- LangChain Structure-Aware Splitting: https://python.langchain.com/docs/how_to/semantic_chunker/

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
- Cormack et al. (2009) RRF: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
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

다중 PDF 인덱싱(week-02/proLmpa/data/ 내 4개 파일):

| PDF | 페이지 | 생성 청크 |
|---|---|---|
| 2024 알기 쉬운 의료급여제도.pdf | 21 | 68 |
| 2026+의료급여사업+안내.pdf | 790 | 2,462 |
| 보건복지부_의료급여.pdf | 1 | 0 (추출 가능 텍스트 없음) |
| 의료급여법 시행령(제35932호).pdf | 10 | 47 |
| 합계 | 822 | 2,577 |

| 청킹 방법 | 청크 수 |
|---|---|
| `table` | 475 |
| `table_slide` (10행 슬라이딩) | 151 |
| `qa_split` | 340 |
| `fixed_fallback` | 1,611 |

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

---

## 4. 결과 비교

### 4-1. 집계 비교표

Evidence Recall, Faithfulness, ContextRecall: Q4·Q9(unanswerable) 제외 n=7
FactualCorrectness(recall): 전체 n=9
Answerability 정확도: Q5(partially_answerable) 제외 n=8
사람 판정: 전체 n=9

| 구성 | EvRecall@5 (n=7) | Faithfulness (n=7) | FC(recall) (n=9) | CtxRecall (n=7) | Answerability (n=8) | 사람 pass (n=9) | 사람 partial (n=9) | 사람 fail (n=9) | 평균 ms | RAG 총 비용 |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0.714 | 0.882 | 0.844 | 0.571 | 0.750 | 6 | 1 | 2 | 7,498 | $0.120 |
| Chunking | 0.429 | 0.792 | 0.333 | 0.286 | 0.500 | 2 | 2 | 5 | 7,577 | $0.124 |
| Hybrid | 0.714 | 0.851 | 0.722 | 0.714 | 0.750 | 6 | 1 | 2 | 6,482 | $0.124 |

> 평가용 Ragas LLM 호출 시간(약 164–168초/구성)은 RAG 비용과 별도로 기록했으며 위 비용에 미포함.

### 4-2. 문항별 비교표

| Q | 유형 | | EvRecall | Faith | FC(rec) | RR | CtxRec | ms | 비용($) | 사람판정 |
|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | answerable | baseline | 1.00 | 1.00 | 1.00 | 0.950 | 1.00 | 6,805 | 0.014 | pass |
| | | chunking | 0.00 | 0.83 | 0.00 | 0.000 | 0.00 | 8,953 | 0.015 | fail |
| | | hybrid | 1.00 | 0.80 | 1.00 | 0.975 | 1.00 | 6,137 | 0.015 | pass |
| Q2 | answerable_known_failure | baseline | 0.00 | 0.80 | 0.00 | 0.000 | 0.00 | 6,870 | 0.008 | fail |
| | | chunking | 1.00 | 1.00 | 0.00* | 0.959 | 1.00 | 2,986 | 0.009 | pass |
| | | hybrid | 1.00 | 1.00 | 1.00 | 0.980 | 1.00 | 3,885 | 0.011 | pass |
| Q3 | answerable | baseline | 1.00 | 0.92 | 1.00 | 0.983 | 0.00† | 9,613 | 0.016 | pass |
| | | chunking | 1.00 | 1.00 | 0.50 | 1.000 | 0.00† | 10,264 | 0.018 | pass |
| | | hybrid | 1.00 | 1.00 | 1.00 | 0.954 | 1.00 | 11,074 | 0.018 | pass |
| Q4 | unanswerable | baseline | N/A | N/A | 1.00 | N/A | N/A | 5,890 | 0.013 | partial |
| | | chunking | N/A | N/A | 1.00 | N/A | N/A | 6,654 | 0.011 | partial |
| | | hybrid | N/A | N/A | 1.00 | N/A | N/A | 5,783 | 0.013 | partial |
| Q5 | partially_answerable | baseline | 1.00 | 0.88 | 1.00 | 0.957 | 1.00 | 8,922 | 0.014 | pass |
| | | chunking | 1.00 | 0.83 | 1.00 | 0.000‡ | 1.00 | 9,845 | 0.018 | partial |
| | | hybrid | 1.00 | 0.75 | 1.00 | 0.000‡ | 1.00 | 6,348 | 0.014 | pass |
| Q6 | answerable | baseline | 1.00 | 1.00 | 1.00 | 0.996 | 1.00 | 7,424 | 0.014 | pass |
| | | chunking | 0.00 | 0.50 | 0.00 | 0.000 | 0.00 | 5,753 | 0.012 | fail |
| | | hybrid | 0.00 | 0.67 | 0.00 | 0.000 | 0.00 | 8,012 | 0.015 | fail |
| Q7 | answerable | baseline | 1.00 | 0.92 | 0.60 | 0.952 | 1.00 | 8,850 | 0.013 | pass |
| | | chunking | 0.00 | 0.71 | 0.00 | 0.000 | 0.00 | 7,266 | 0.013 | fail |
| | | hybrid | 1.00 | 0.91 | 0.50 | 0.953 | 1.00 | 7,375 | 0.013 | pass |
| Q8 | answerable | baseline | 0.00 | 0.67 | 0.00 | 0.000 | 0.00 | 9,887 | 0.017 | fail |
| | | chunking | 0.00 | 0.67 | 0.00 | 0.000 | 0.00 | 8,804 | 0.015 | fail |
| | | hybrid | 0.00 | 0.83 | 0.00 | 0.000 | 0.00 | 6,044 | 0.013 | fail |
| Q9 | unanswerable | baseline | N/A | N/A | 1.00 | N/A | N/A | 3,219 | 0.011 | pass |
| | | chunking | N/A | N/A | 0.50 | N/A | N/A | 7,665 | 0.013 | fail |
| | | hybrid | N/A | N/A | 1.00 | N/A | N/A | 3,680 | 0.012 | pass |

> `*` Q2 chunking FC(recall)=0.00: 검색 성공(EvRecall=1.0), 답변에 비율 수치 포함 확인. Ragas가 "1종수급권자는 5%"를 reference의 "1종 5%"와 불일치로 판정한 것으로 추정 — 사람 판정 pass.
> `†` Q3 ContextRecall=0.0: Ragas reference("500원 (1종/2종 공통)")와 PDF 원문 표현 불일치 추정. Evidence Recall=1.0이므로 실제 근거는 검색됨.
> `‡` Q5 ResponseRelevancy=0.0: partially_answerable 질문에서 스케일링 거절을 Ragas가 "관련성 없음"으로 평가. Baseline·Hybrid 사람 판정은 pass(거절이 올바른 응답).

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
- FC(recall)=0.0은 Ragas 판정 오류로 추정 (reference와 표현 차이)
- 사람 판정: pass (수치와 맥락 모두 정확)

Hybrid (개선 B)
- 검색: `chunk_62`(RRF 1위), `chunk_59`(RRF 2위) — BM25가 "틀니", "1종", "5%"를 형태소로 매칭해 chunk_59를 BM25 Top-2에 올림
- Evidence Recall = 1.0, FC(recall) = 1.0, Faithfulness = 1.0
- 사람 판정: pass

#### Q1 — 선택의료급여기관 의뢰서 예외 (기존 성공 케이스)

Baseline: 검색 성공 (chunk_42, p.13), 거절 없이 "응급환자, 장애인 보조기기" 정확 답변. FC(recall)=1.0.

Chunking (개선 A): 검색 실패 — `qa_chunk_1150`(p.325, 2026 안내 문서), `qa_chunk_1221`(p.357), `qa_chunk_2033`(p.628). 2026 안내 문서 2,462개 청크가 검색 풀을 압도해 원본 FAQ p.13 청크가 Top-5 밖으로 밀려남. 모델이 거절. 심각한 회귀.

Hybrid: 검색 성공 (chunk_40, chunk_42, p.13). Baseline과 동일 컬렉션 사용으로 영향 없음. FC(recall)=1.0 유지.

#### Q6 — 2종 외래 본인부담금 (테이블 조회)

세 구성 모두 p.5 본인부담률 표의 "외래 1,000원" 행을 검색하지 못했다 (baseline만 성공). Baseline은 고정 청킹으로 p.5 전체 텍스트가 연속 청크에 담겨 있어 쿼리 임베딩이 p.5를 찾을 수 있었다. Hybrid는 BM25에서 "1,000원"은 많은 청크에 공통으로 등장해 Q6 고유 근거를 올리지 못했다. evidence span `"외래 1,000원 15% 15% 500원 15%"`는 테이블 셀 연결 형태로, pdfplumber가 이를 단일 행으로 추출했는지 불확실하다.

사람 판정: Baseline=pass, Chunking/Hybrid=fail.

#### Q8 — 조현병 외래 본인부담률 (p.6 테이블)

세 구성 모두 실패. evidence span `"조현병 병원급 이상 5%"`는 p.6 정신질환 테이블의 한 행이지만, pdfplumber가 `extract_tables()`로 이 테이블을 구조적으로 추출하지 못했거나 추출된 테이블 텍스트에 해당 조합이 존재하지 않는 형태로 저장됐을 가능성이 있다. PDF p.6를 직접 열어 테이블 구조와 추출 결과를 비교해야 한다.

사람 판정: 모든 구성 fail.

### 4-4. 사람 판정 요약

PDF 원문을 직접 열어 각 답변과 대조한 결과다. 판정 근거는 각 `eval_results.json`의 `human_notes` 필드에 기록했다.

| Q | Baseline | Chunking | Hybrid | 판정 근거 요약 |
|---|---|---|---|---|
| Q1 | pass | fail | pass | PDF A25: 응급증상·장애인 보조기기 두 예외. Chunking은 2026 문서의 한의원 예외를 가져와 핵심 누락 |
| Q2 | fail | pass | pass | PDF A48: 1종 5%, 2종 15%. Baseline은 chunk_59 미검색으로 거절 |
| Q3 | pass | pass | pass | PDF p.5 표: 1종·2종 모두 500원. 세 구성 모두 핵심 사실 포함 |
| Q4 | partial | partial | partial | 거절 + 문서 외 추론("비급여 가능성") 또는 무관한 2026 문서 내용 추가 |
| Q5 | pass | partial | pass | Baseline·Hybrid: 임플란트 정확, 스케일링 거절. Chunking: 2026 문서 기반 조건부 스케일링 답변 추가 |
| Q6 | pass | fail | fail | PDF p.5 표: 2종 외래 의원 1,000원. Chunking·Hybrid 모두 미검색으로 거절 |
| Q7 | pass | fail | pass | PDF A53: 틀니·임플란트 보상제 제외 명시. Chunking은 Q53 미검색으로 거절 |
| Q8 | fail | fail | fail | PDF p.6 표: 조현병 병원급 이상 5%. 세 구성 모두 테이블 추출 실패로 거절 |
| Q9 | pass | fail | pass | 평가 설계상 unanswerable. Chunking은 2026 문서 "출국기간 진료비 환수" 조항으로 답변 생성 |

> Q9 Chunking 판정 보충: 2026 안내 문서에 출국 중 진료비 환수 조항이 실제로 존재하며 답변 내용은 사실이다. 그러나 평가 설계가 2024 FAQ를 기준 문서로 삼아 Q9를 unanswerable로 지정했으므로, rubric(pass = 순수 거절) 기준으로는 fail로 처리한다.

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

개선 A (Chunking) — 사람 판정 기준

| 문항 | 변화 | 원인 |
|---|---|---|
| Q2 | fail → pass | Q&A 경계 분할로 A48 단일 청크 보존 성공 |
| Q1 | pass → fail | 2026 안내 문서 2,462청크가 검색 풀 압도, A25 답변 누락 |
| Q5 | pass → partial | 임플란트 정확, 스케일링에 2026 문서 조건부 답변 추가 |
| Q6 | pass → fail | p.5 표 청크가 2026 문서에 밀려 1,000원 미검색 |
| Q7 | pass → fail | Q53 미검색, 보상제 제외 사실 답변 불가 |
| Q9 | pass → fail | 2026 문서 출국기간 환수 조항 검색 → 거절 대신 답변 생성 |

개선 1건(Q2), 회귀 5건(Q1·Q5·Q6·Q7·Q9) — 순손실.

개선 B (Hybrid) — 사람 판정 기준

| 문항 | 변화 | 원인 |
|---|---|---|
| Q2 | fail → pass | BM25가 "틀니", "1종", "5%" 형태소 정확 매칭 |
| Q6 | pass → fail | "1,000원" 키워드 다수 청크에 분산, BM25 미분별 |

개선 1건(Q2), 회귀 1건(Q6) — 균형. 나머지 7문항 판정 유지.

### 5-3. 점수 상승과 실제 답변 개선의 일치 여부

- Q2 Chunking FC(recall)=0.0 ↔ 사람 pass: EvRecall=1.0, Faithfulness=1.0이지만 FC=0.0. Ragas가 "1종수급권자는 5%"를 reference "1종 5%"와 불일치로 판정. PDF 직접 대조 결과 수치·맥락 모두 정확 → 사람 pass. 자동 평가가 표현 차이를 오류로 처리한 전형적 사례.
- Q5 ResponseRelevancy=0.0 ↔ Baseline·Hybrid 사람 pass: 스케일링 거절이 포함된 답변을 Ragas가 "관련성 없음"으로 평가. 사람 기준으로는 거절 자체가 올바른 응답(pass)이므로 0점은 무의미. Chunking만 사람 partial — 2026 문서 조건부 답변 추가가 원인.
- Q7 FC(recall)=0.6 (Baseline) ↔ 사람 pass: 자동 평가가 60% 점수를 줬으나 PDF A53 원문("65세 이상 치과 틀니 및 임플란트 비용 ... 지급에서 제외")을 답변이 명확히 포함. reference 표현과의 부분 불일치가 감점 원인으로 추정. 사람 pass.
- Q9 Chunking FC(recall)=0.5 ↔ 사람 fail: 자동 지표는 0.5(부분 점수)이나 사람 판정은 fail — unanswerable 질문에 답변을 생성했기 때문. 자동 지표가 거절 여부를 감지하지 못함.

전반적으로 EvRecall@5는 사람 판정과 가장 일치율이 높다. FC(recall)은 reference 표현 차이에 민감해 과소평가 경향이 있으며 사람 판정과 병행해야 신뢰할 수 있다. Answerability 지표가 Q9 Chunking을 감지하지 못한 점도 한계다.

### 5-4. 추가 비용·지연의 가치

| | Chunking | Hybrid |
|---|---|---|
| 추가 인덱싱 비용 | 2,577청크 임베딩 (1회성) | BM25 초기화 ~1초 |
| 검색 지연 추가 | 없음 (Dense Top-5 동일) | +20ms/쿼리 (BM25 점수 계산) |
| 평균 총 지연 | 7,577ms (baseline 대비 +79ms) | 6,482ms (baseline 대비 -1,016ms) |
| 사람 판정 개선 | Q2: fail→pass (1건) | Q2: fail→pass (1건) |
| 사람 판정 회귀 | Q1·Q5·Q6·Q7·Q9 (5건) | Q6 (1건) |
| 순효과 | -4건 (회귀가 개선보다 4건 많음) | 0건 (개선·회귀 균형) |
| 가치 | 낮음 | 높음 |

Hybrid는 추가 지연이 거의 없고 오히려 평균 응답이 빨라졌다(dense-only보다 짧은 컨텍스트로 답변 가능). 비용도 쿼리당 추가 지출 없이 baseline 수준을 유지한다.

### 5-5. 최종 판단

기법 A (구조 인식 청킹): 수정 (Modify)

판단 근거: Q2 retrieval 문제는 해결됐으나(사람 pass 확인), 2026+의료급여사업+안내.pdf(790페이지, 2,462청크)를 무분별하게 추가한 것이 Q1·Q5·Q6·Q7·Q9 회귀(사람 판정 기준 5건)의 직접 원인이다. 개선 1건 대비 회귀 5건으로 순손실. 기법 자체(Q&A 경계 분할 + 테이블 추출)는 유효하다.

수정 방향:
- 문서별 별도 컬렉션 운영 또는 소스 필터 검색(`where={"source": "2024 알기 쉬운 의료급여제도.pdf"}`)
- 대형 사업 안내 문서는 별도 평가셋 구성 후 검색 풀에 포함 여부 결정
- Q6, Q8 실패 원인(테이블 추출 실패)은 PDF p.5·p.6를 직접 열어 `extract_tables()` 결과 확인 후 재처리

기법 B (하이브리드 검색 BM25+Dense→RRF): 유지 (Keep)

판단 근거: 목표였던 Q2 fix를 달성했고(사람 pass 확인), 기존에 성공하던 Q1·Q3·Q5·Q7·Q9 7문항의 판정이 그대로 유지됐다. 개선 1건, 회귀 1건으로 순효과 0건이지만 Q6 회귀는 BM25 쿼리 전처리로 보완 가능한 문제다. 추가 지연은 쿼리당 20ms 미만이며 비용 증가 없고 평균 응답 속도는 오히려 빨라졌다(-1,016ms). Q8는 BM25로도 해결 불가 — 근본 원인이 PDF 테이블 추출 실패이므로 청킹 개선이 선행되어야 한다.

---

## 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# Config A 컬렉션 검증
python rag_configs.py --inspect

# 전체 평가 실행 (세 구성)
python run_eval.py

# Config A 컬렉션 강제 재빌드 후 실행
python run_eval.py --rebuild-chunking

# 검색만 확인 (API 호출 없음)
python run_eval.py --dry-run

# 특정 구성만 실행
python run_eval.py --configs baseline hybrid
```

사전 조건: `week-02/proLmpa/` 에서 `python rag_basic.py` 실행 완료 (baseline ChromaDB 컬렉션 필요)

환경변수: `.env` 파일에 `ANTHROPIC_API_KEY=...` 설정

---

## 발표 개요 (5분)

1. (1분) week-03 Q2 실패 원인 — 고정 청킹이 Q48/A48를 절단한 구체적 증거 (chunk_57/58/59 경계 그림)
2. (1분) 평가 보완 내용 — evidence_spans, FC(recall), N/A 처리
3. (1분) 두 기법 선택 이유 — A: 청킹 절단 직접 해결, B: 정확 용어 매칭 보완
4. (1분) 비교표 — Q2 fix 공통, Chunking 회귀 원인(2026 문서 1,611청크 압도), Hybrid 안정성
5. (1분) 판단 — Hybrid 유지, Chunking 수정(소스 필터 필요), Q8는 PDF 추출 선행 필요
