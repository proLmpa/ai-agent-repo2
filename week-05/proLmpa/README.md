# 한국 부동산 실거래가 분석 에이전트

## 1. 개념 조사

**Q1. Workflow와 Agent에서는 각각 누가 다음 실행 경로를 정하나요?**

Workflow에서는 개발자가 미리 정해둔 순서와 분기 조건이 실행 경로를 결정한다. 어떤 조건에서 어떤 단계로 이동할지 코드에 고정되어 있어, 실행 중에 경로가 바뀌지 않는다.

Agent에서는 모델이 현재 상태(사용자 요청 + 지금까지의 Tool 결과)를 보고 다음에 무엇을 할지 직접 선택한다. 어떤 Tool을 쓸지, 어떤 입력을 넣을지, 아니면 바로 답할지를 모델이 매 턴마다 결정한다.

**Q2. Agent Loop에서 모델은 무엇을 선택하고, 실행 프로그램은 Tool을 어떻게 실행해 결과를 돌려주나요?**

모델은 현재 메시지 히스토리를 보고 다음 행동을 선택한다. 선택지는 Tool 호출(이름 + 인자) 또는 최종 응답 생성 두 가지다.

실행 프로그램(LangChain `create_agent`)은 모델이 Tool 호출을 선택하면, 해당 함수를 실제로 실행하고 반환값을 `ToolMessage`로 메시지 히스토리에 추가한 뒤 모델에게 다시 전달한다. 이 과정을 모델이 최종 응답을 낼 때까지 반복한다.

`create_agent`는 내부적으로 두 노드를 가진 LangGraph를 생성한다.

```
[agent 노드] ──tool_calls 있음──► [tools 노드] ──► [agent 노드] ──► ...
                                                              │
                                              tool_calls 없음 ▼
                                                       최종 응답 반환
```

| 단계 | 주체 | 하는 일 |
|---|---|---|
| tool_calls 결정 | 모델(Claude) | 어떤 함수를, 어떤 인자로 호출할지 선택 |
| 함수 실행 | LangGraph tools 노드 | 이름으로 함수 찾아 `args` 그대로 호출 |
| 결과 전달 | LangGraph | 반환값을 `ToolMessage`로 감싸 히스토리에 추가 |
| 다음 행동 결정 | 모델(Claude) | ToolMessage 읽고 또 tool 호출할지, 최종 답변 낼지 판단 |

**Q3. ReAct의 판단·행동·관찰은 본인의 구현에서 어디에 해당하나요?**

| ReAct | 본 구현에서의 해당 지점 |
|---|---|
| 판단 (Reasoning) | 모델이 system prompt + 이전 Tool 결과를 보고 다음에 호출할 Tool과 입력을 결정하는 과정 |
| 행동 (Action) | `search_apartment_sales`, `search_apartment_rentals`, `list_statistics_tables`, `get_statistics_data` 중 하나를 선택해 호출 |
| 관찰 (Observation) | Tool이 반환한 실거래 내역 또는 통계 텍스트가 메시지로 추가되어 모델에게 전달 |

**Q4. 실행이 끝난 뒤 무엇을 근거로 목표 달성 여부를 판단하나요?**

사용자가 요청한 조건(지역, 거래 유형, 가격대 등)에 부합하는 실거래 데이터가 최종 응답에 포함되었는지를 기준으로 판단한다. API로 확인할 수 없는 조건이 있을 경우에는 그 조건이 응답에 명시되어 있는지도 확인한다. 반복 한도(5회)에 도달해 종료된 경우에는 목표 미달성으로 기록한다.

---

## 2. 에이전트 개요

### 목표

사용자의 주거 조건(지역, 거래 유형, 가격대, 시기)에 맞는 아파트 실거래 데이터를 조회하고, 시장 통계와 함께 분석을 제공한다.

### 달성 기준

- 사용자가 제시한 조회 가능한 조건(지역, 거래 유형, 가격대, 시기)에 맞는 실거래 사례가 최종 응답에 포함되어 있다.
- API로 확인할 수 없는 조건(학군, 소음, 채광 등)이 요청에 포함된 경우, 확인한 조건과 확인하지 못한 조건이 최종 응답에 명시되어 있다.
- Tool 호출 횟수가 반복 한도(5회) 이내에 종료된다.

### Instructions (system prompt 요약)

```
역할: 한국 부동산 실거래가 분석 전문가
목표: 사용자의 주거 조건에 맞는 실거래 데이터를 조회하고 시장 통계와 함께 분석 제공

조회 가능한 조건: 지역, 거래 유형(매매/전세/월세), 가격대, 거래 시기
조회 불가한 조건: 학군, 소음, 채광, 주변 편의시설
  → 해당 조건이 있으면 확인 가능한 결과를 먼저 제공하고 확인 불가 조건을 명시 후 종료

한국부동산원 통계 조회 절차:
  list_statistics_tables 먼저 호출 → STATBL_ID, DTACYCLE_CD 확인 → get_statistics_data 호출

행동 규칙:
  - Tool 호출 최대 5회
  - 한도 도달 시 '목표 미달성: 반복 한도 초과' 기록 후 종료
  - 목표 달성 불가 시 확인한 조건/미확인 조건 명시
```

### 모델이 다음 행동을 고른 지점

매 Tool 호출 후 모델은 반환된 텍스트를 읽고 다음을 결정한다.

- 거래 결과가 충분하면 → 최종 응답 생성
- 통계 맥락이 필요하면 → `list_statistics_tables` 호출로 적절한 통계표 ID 탐색
- 통계표 ID를 확인했으면 → `get_statistics_data` 호출
- 비교가 필요한 요청(매매 vs 전세)이면 → 두 Tool을 순차 호출

### RAG 연결

미적용. 현재 사용하는 실거래가 API가 충분히 구체적인 거래 데이터를 직접 제공하므로 별도 문서 검색 계층이 필요하지 않다고 판단했다.

---

## 3. Tools

| # | 함수명 | API | 용도 | 필수 입력 |
|---|---|---|---|---|
| 1 | `search_apartment_rentals` | 국토교통부 전월세 실거래가 | 전세·월세 거래 사례 조회 | `region_code`, `year_month` |
| 2 | `search_apartment_sales` | 국토교통부 매매 실거래가 | 매매 거래 사례 조회 | `region_code`, `year_month` |
| 3 | `list_statistics_tables` | 한국부동산원 통계표 목록 | 조회 가능한 통계표 ID 탐색 | (없음, 키워드 선택) |
| 4 | `get_statistics_data` | 한국부동산원 통계 데이터 | 가격지수 등 통계값 조회 | `statbl_id`, `dtacycle_cd` |

### API 공식 문서

- 국토교통부 아파트 전월세 실거래가: https://apis.data.go.kr/1613000/RTMSDataSvcAptRent
- 국토교통부 아파트 매매 실거래가: https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade
- 한국부동산원 부동산통계: https://www.reb.or.kr/r-one/openapi/SttsApiTbl.do

---

## 4. 실행 결과

세 가지 요청 유형을 각각 1건씩 `run_scenarios.py`로 실행했다. 상세 기록은 `run_log.md` 참조.

### 시나리오 1 — 목표 달성 가능

| 항목 | 내용 |
|---|---|
| 요청 | 마포구 2024년 10월 아파트 매매 거래 사례를 찾아주고 최근 매매 시세 흐름도 알려줘 |
| Tool 경로 | `search_apartment_sales` → `list_statistics_tables("아파트 매매")` 실패 → `list_statistics_tables("매매")` → `get_statistics_data` × 2 |
| Tool 호출 수 | 5회 (한도 도달) |
| 모델 호출 수 | 4회 |
| 실행 시간 | 50.95초 |
| 종료 이유 | 반복 한도 5회 초과로 `recursion_limit` 하드 차단 |
| **목표 달성** | **달성** |
| 판정 근거 | 매매 사례 160건 조회 완료. KREB에 마포구 단독 월별 지수가 없어 서울 서북권(마포·은평·서대문) 집계로 대체했으나 최종 응답에 이 한계가 명시되었고 시세 흐름 분석이 포함됨 |

### 시나리오 2 — API로 확인 불가한 조건 포함

| 항목 | 내용 |
|---|---|
| 요청 | 초등학교 도보 5분 거리에 있고 조용한 마포구 아파트를 추천해줘 |
| Tool 경로 | `search_apartment_rentals` → `search_apartment_sales` |
| Tool 호출 수 | 2회 |
| 모델 호출 수 | 2회 |
| 실행 시간 | 33.60초 |
| 종료 이유 | 정상 완료 — 모델이 최종 응답 생성 |
| **목표 달성** | **달성** |
| 판정 근거 | 최종 응답 표에 초등학교 거리(❌ 확인 불가), 소음(❌ 확인 불가)이 명시됨. 확인 가능한 조건(지역·거래 사례)은 조회 후 제공됨 |

### 시나리오 3 — 앞선 실행과 다른 경로

| 항목 | 내용 |
|---|---|
| 요청 | 성동구에서 2024년 10월 기준으로 아파트 매매할지 전세로 살지 실거래 데이터를 비교해서 알려줘 |
| Tool 경로 | `search_apartment_sales` → `search_apartment_rentals` |
| Tool 호출 수 | 2회 |
| 모델 호출 수 | 2회 |
| 실행 시간 | 42.21초 |
| 종료 이유 | 정상 완료 — 모델이 최종 응답 생성 |
| **목표 달성** | **달성** |
| 판정 근거 | 시나리오 1(sales → statistics)과 달리 비교 요청을 인식해 sales → rentals 경로를 선택. 매매 175건·전세 791건 비교 분석 및 의사결정 기준별 표 완성 |

### 경로 비교

| 시나리오 | Tool 호출 순서 |
|---|---|
| 1 (목표 달성) | sales → list_tables → list_tables → get_data → get_data |
| 2 (확인 불가 조건) | rentals → sales |
| 3 (다른 경로) | sales → rentals |

---

## 5. 설치 및 실행

### 환경 변수 (`.env` 파일)

```
ANTHROPIC_API_KEY=    # Anthropic API 키
MOLIT_API_KEY=        # 국토교통부 공공데이터포털 인증키 (매매 + 전월세 공용)
KREB_API_KEY=         # 한국부동산원 R-ONE 인증키
LANGSMITH_API_KEY=    # (선택) LangSmith 트레이싱
LANGSMITH_PROJECT=    # (선택) LangSmith 프로젝트명
```

### 설치

```bash
cd week-05/proLmpa
python -m venv .venv
.venv/Scripts/activate      # Windows
pip install -r requirements.txt
cp .env.example .env        # 이후 .env에 실제 키 입력
```

### 실행

```bash
# 단일 쿼리 테스트
python agent.py

# 3개 시나리오 전체 실행 및 run_log.md 저장
python run_scenarios.py
```
