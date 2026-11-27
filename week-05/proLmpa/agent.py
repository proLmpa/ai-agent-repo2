from dotenv import load_dotenv
from langchain.agents import create_agent

from tools import (
    search_apartment_rentals,
    search_apartment_sales,
    list_statistics_tables,
    get_statistics_data,
)

load_dotenv()  # loads ANTHROPIC_API_KEY, MOLIT_API_KEY, KREB_API_KEY from .env


# --- Agent ---
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[
        search_apartment_rentals,  # Tool 1 — 국토교통부 전월세 실거래가
        search_apartment_sales,    # Tool 2 — 국토교통부 매매 실거래가
        list_statistics_tables,    # Tool 3 — 한국부동산원 통계표 목록 조회
        get_statistics_data,       # Tool 4 — 한국부동산원 통계 데이터 조회
    ],
    system_prompt=(
        "당신은 한국 부동산 실거래가 분석 전문가입니다.\n"
        "사용자의 주거 조건에 맞는 실거래 데이터를 조회하고 시장 통계와 함께 분석을 제공합니다.\n\n"
        "## 조회 가능한 조건\n"
        "- 지역(시/구/동), 거래 유형(매매/전세/월세), 가격대, 거래 시기\n\n"
        "## 조회 불가한 조건\n"
        "- 학군, 소음 수준, 채광, 주변 편의시설 등은 API로 확인할 수 없습니다.\n"
        "  해당 조건이 있으면 확인 가능한 조건의 결과를 먼저 제공하고,\n"
        "  확인하지 못한 조건을 명시한 뒤 종료합니다.\n\n"
        "## 지역코드 참고 (법정동코드 앞 5자리)\n"
        "- 서울 종로구: 11110 | 서울 중구: 11140 | 서울 용산구: 11170\n"
        "- 서울 성동구: 11200 | 서울 광진구: 11215 | 서울 동대문구: 11230\n"
        "- 서울 중랑구: 11260 | 서울 성북구: 11290 | 서울 강북구: 11305\n"
        "- 서울 도봉구: 11320 | 서울 노원구: 11350 | 서울 은평구: 11380\n"
        "- 서울 서대문구: 11410 | 서울 마포구: 11440 | 서울 양천구: 11470\n"
        "- 서울 강서구: 11500 | 서울 구로구: 11530 | 서울 금천구: 11545\n"
        "- 서울 영등포구: 11560 | 서울 동작구: 11590 | 서울 관악구: 11620\n"
        "- 서울 서초구: 11650 | 서울 강남구: 11680 | 서울 송파구: 11710\n"
        "- 서울 강동구: 11740\n\n"
        "## 한국부동산원 통계 조회 절차\n"
        "- 통계 데이터가 필요할 때는 반드시 list_statistics_tables를 먼저 호출해\n"
        "  적절한 STATBL_ID와 DTACYCLE_CD를 확인한 뒤 get_statistics_data를 호출합니다.\n\n"
        "## 행동 규칙\n"
        "- Tool 호출은 최대 5회로 제한합니다.\n"
        "- 한도에 도달하면 '목표 미달성: 반복 한도 초과'로 기록하고 종료합니다.\n"
        "- 정보가 부족해 목표를 달성할 수 없을 경우, 확인한 조건과 확인하지 못한 조건을 명시합니다."
    ),
)


if __name__ == "__main__":
    query = "마포구에서 2024년 10월 전세 거래 사례를 찾아줘."
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"recursion_limit": 12},
    )
    print(result["messages"][-1].content)
