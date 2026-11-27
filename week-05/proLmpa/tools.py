"""
API tool functions for the Korean real estate agent.

Each function is registered as a LangChain tool via its docstring.
API keys are loaded from environment variables — never hardcoded here.
"""

import os
import xml.etree.ElementTree as ET

import requests


# ---------------------------------------------------------------------------
# Tool 1 — 아파트 전월세 실거래가 (국토교통부)
# Endpoint: https://apis.data.go.kr/1613000/RTMSDataSvcAptRent/getRTMSDataSvcAptRent
# ---------------------------------------------------------------------------

def search_apartment_rentals(
    region_code: str,
    year_month: str,
    deal_type: str = "전체",
    num_of_rows: int = 20,
) -> str:
    """아파트 전월세 실거래가를 조회합니다.

    국토교통부 실거래가 공개시스템 API를 사용해 특정 지역과 계약월의
    아파트 전세·월세 실거래 내역을 반환합니다.

    Args:
        region_code: 법정동코드 앞 5자리 (예: 서울 종로구 → 11110,
                     서울 마포구 → 11440, 서울 성동구 → 11200)
        year_month:  계약년월 6자리 (예: 202410)
        deal_type:   "전세", "월세", "전체" 중 하나 (기본값: "전체")
        num_of_rows: 최대 조회 건수 (기본값: 20)

    Returns:
        조회 결과 텍스트. 거래 없으면 "조회된 거래가 없습니다." 반환.
    """
    api_key = os.environ.get("MOLIT_API_KEY", "")
    url = (
        "https://apis.data.go.kr/1613000/RTMSDataSvcAptRent"
        "/getRTMSDataSvcAptRent"
    )
    params = {
        "serviceKey": api_key,
        "LAWD_CD": region_code,
        "DEAL_YMD": year_month,
        "numOfRows": num_of_rows,
        "pageNo": 1,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"API 호출 오류: {e}"

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        return f"XML 파싱 오류: {e}\n응답 내용: {resp.text[:300]}"

    # Check API-level error code
    result_code = root.findtext(".//resultCode", "")
    result_msg = root.findtext(".//resultMsg", "")
    if result_code not in ("", "000", "00", "0000", "OK"):
        return f"API 오류 (코드 {result_code}): {result_msg}"

    items = root.findall(".//item")
    if not items:
        return "조회된 거래가 없습니다."

    rows = []
    for item in items:
        monthly_rent_raw = item.findtext("monthlyRent", "0").strip()
        monthly_rent = int(monthly_rent_raw.replace(",", "") or "0")
        inferred_type = "월세" if monthly_rent > 0 else "전세"

        # Client-side filter
        if deal_type != "전체" and inferred_type != deal_type:
            continue

        rows.append({
            "법정동": item.findtext("umdNm", "").strip(),
            "아파트명": item.findtext("aptNm", "").strip(),
            "전용면적(㎡)": item.findtext("excluUseAr", "").strip(),
            "층": item.findtext("floor", "").strip(),
            "계약일": (
                f"{item.findtext('dealYear','')}.{item.findtext('dealMonth','').zfill(2)}"
                f".{item.findtext('dealDay','').zfill(2)}"
            ),
            "거래유형": inferred_type,
            "보증금(만원)": item.findtext("deposit", "").strip(),
            "월세(만원)": str(monthly_rent) if monthly_rent > 0 else "-",
            "건축년도": item.findtext("buildYear", "").strip(),
        })

    if not rows:
        return f"조회 결과는 있으나 '{deal_type}' 거래 유형에 해당하는 건이 없습니다."

    total = root.findtext(".//totalCount", "?")
    lines = [
        f"[아파트 전월세 실거래가] 지역코드={region_code} / 계약월={year_month}"
        f" / 조건={deal_type} / 전체{total}건 중 {len(rows)}건 표시\n"
    ]
    for r in rows:
        lines.append(
            f"- {r['법정동']} {r['아파트명']} {r['전용면적(㎡)']}㎡ {r['층']}층 "
            f"({r['건축년도']}년) | {r['거래유형']} | 보증금 {r['보증금(만원)']}만원"
            + (f" / 월세 {r['월세(만원)']}만원" if r["월세(만원)"] != "-" else "")
            + f" | 계약일 {r['계약일']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2 — 아파트 매매 실거래가 (국토교통부)
# Endpoint: https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade
# ---------------------------------------------------------------------------

def search_apartment_sales(
    region_code: str,
    year_month: str,
    num_of_rows: int = 20,
) -> str:
    """아파트 매매 실거래가를 조회합니다.

    국토교통부 실거래가 공개시스템 API를 사용해 특정 지역과 계약월의
    아파트 매매 실거래 내역을 반환합니다.

    Args:
        region_code: 법정동코드 앞 5자리 (예: 서울 종로구 → 11110,
                     서울 마포구 → 11440, 서울 강남구 → 11680)
        year_month:  계약년월 6자리 (예: 202410)
        num_of_rows: 최대 조회 건수 (기본값: 20)

    Returns:
        조회 결과 텍스트. 거래 없으면 "조회된 거래가 없습니다." 반환.
    """
    api_key = os.environ.get("MOLIT_API_KEY", "")
    url = (
        "https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade"
        "/getRTMSDataSvcAptTrade"
    )
    params = {
        "serviceKey": api_key,
        "LAWD_CD": region_code,
        "DEAL_YMD": year_month,
        "numOfRows": num_of_rows,
        "pageNo": 1,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"API 호출 오류: {e}"

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        return f"XML 파싱 오류: {e}\n응답 내용: {resp.text[:300]}"

    # Check API-level error code
    result_code = root.findtext(".//resultCode", "")
    result_msg = root.findtext(".//resultMsg", "")
    if result_code not in ("", "000", "00", "0000", "OK"):
        return f"API 오류 (코드 {result_code}): {result_msg}"

    items = root.findall(".//item")
    if not items:
        return "조회된 거래가 없습니다."

    rows = []
    for item in items:
        # 해제된 계약(cdealType이 있으면) 제외
        if item.findtext("cdealType", "").strip():
            continue

        rows.append({
            "법정동": item.findtext("umdNm", "").strip(),
            "아파트명": item.findtext("aptNm", "").strip(),
            "전용면적(㎡)": item.findtext("excluUseAr", "").strip(),
            "층": item.findtext("floor", "").strip(),
            "계약일": (
                f"{item.findtext('dealYear','')}.{item.findtext('dealMonth','').zfill(2)}"
                f".{item.findtext('dealDay','').zfill(2)}"
            ),
            "거래금액(만원)": item.findtext("dealAmount", "").strip(),
            "거래유형": item.findtext("dealingGbn", "").strip(),
            "건축년도": item.findtext("buildYear", "").strip(),
            "매도자": item.findtext("slerGbn", "").strip(),
            "매수자": item.findtext("buyerGbn", "").strip(),
        })

    if not rows:
        return "조회 결과는 있으나 유효한 매매 거래(해제 제외)가 없습니다."

    total = root.findtext(".//totalCount", "?")
    lines = [
        f"[아파트 매매 실거래가] 지역코드={region_code} / 계약월={year_month}"
        f" / 전체{total}건 중 {len(rows)}건 표시 (해제 거래 제외)\n"
    ]
    for r in rows:
        lines.append(
            f"- {r['법정동']} {r['아파트명']} {r['전용면적(㎡)']}㎡ {r['층']}층 "
            f"({r['건축년도']}년) | {r['거래금액(만원)']}만원 | {r['거래유형']} "
            f"| 매도 {r['매도자']} → 매수 {r['매수자']} | 계약일 {r['계약일']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3 — 한국부동산원 통계표 목록 조회
# Endpoint: https://www.reb.or.kr/r-one/openapi/SttsApiTbl.do
# ---------------------------------------------------------------------------

def list_statistics_tables(keyword: str = "") -> str:
    """한국부동산원에서 제공하는 부동산 통계표 목록을 조회합니다.

    통계 데이터를 조회하기 전에 반드시 이 Tool을 먼저 호출해 적절한
    통계표 ID(STATBL_ID)와 주기코드(DTACYCLE_CD)를 확인합니다.

    Args:
        keyword: 통계표명에서 검색할 키워드 (예: "아파트", "매매", "전세").
                 빈 문자열이면 전체 목록을 반환합니다.

    Returns:
        통계표 ID, 통계표명, 주기코드, 주기명, 자료 시작·종료년도를 포함한
        목록 텍스트. 이후 get_statistics_data 호출 시 STATBL_ID와
        DTACYCLE_CD를 사용합니다.
    """
    api_key = os.environ.get("KREB_API_KEY", "")
    url = "https://www.reb.or.kr/r-one/openapi/SttsApiTbl.do"
    params = {
        "KEY": api_key,
        "Type": "xml",
        "pIndex": 1,
        "pSize": 100,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"API 호출 오류: {e}"

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        return f"XML 파싱 오류: {e}\n응답 내용: {resp.text[:300]}"

    # Check API-level error code
    code = root.findtext(".//CODE", root.findtext(".//code", ""))
    if code and code not in ("000", "INFO-000"):
        msg = root.findtext(".//MESSAGE", root.findtext(".//message", ""))
        return f"API 오류 (코드 {code}): {msg}"

    items = root.findall(".//row")
    if not items:
        return "조회된 통계표가 없습니다."

    # Keyword filter (client-side)
    if keyword:
        items = [
            it for it in items
            if keyword in (it.findtext("STATBL_NM", "") or "")
        ]
    if not items:
        return f"'{keyword}'에 해당하는 통계표가 없습니다."

    lines = [f"[한국부동산원 통계표 목록] 키워드='{keyword}' / {len(items)}건\n"]
    for it in items:
        lines.append(
            f"- STATBL_ID={it.findtext('STATBL_ID','').strip()} "
            f"| {it.findtext('STATBL_NM','').strip()} "
            f"| 주기: {it.findtext('DTACYCLE_NM','').strip()} ({it.findtext('DTACYCLE_CD','').strip()}) "
            f"| 기간: {it.findtext('DATA_START_YY','').strip()}~{it.findtext('DATA_END_YY','').strip()}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4 — 한국부동산원 통계 데이터 조회
# Endpoint: https://www.reb.or.kr/r-one/openapi/SttsApiTblData.do
# ---------------------------------------------------------------------------

def get_statistics_data(
    statbl_id: str,
    dtacycle_cd: str,
    start_wrttime: str = "",
    end_wrttime: str = "",
) -> str:
    """한국부동산원 특정 통계표의 실제 데이터를 조회합니다.

    list_statistics_tables로 확인한 STATBL_ID와 DTACYCLE_CD를 사용합니다.
    결과에 포함된 CLS_FULLNM(분류 전체명)을 보고 필요한 지역·항목을 직접 판단하십시오.

    Args:
        statbl_id:     통계표 ID (list_statistics_tables 결과에서 확인)
        dtacycle_cd:   주기코드 (예: "MM" 월간, "QY" 분기, "YY" 연간)
        start_wrttime: 자료작성 시점 시작 (예: "202401").
                       빈 문자열이면 가장 최근 데이터를 반환합니다.
        end_wrttime:   자료작성 시점 종료. 빈 문자열이면 start_wrttime과 동일하게 적용됩니다.

    Returns:
        분류 전체명(CLS_FULLNM), 항목명, 단위, 값, 작성 시점을 포함한 데이터 텍스트.
        반환된 분류명을 보고 원하는 지역 데이터를 선택하십시오.
    """
    api_key = os.environ.get("KREB_API_KEY", "")
    url = "https://www.reb.or.kr/r-one/openapi/SttsApiTblData.do"
    params = {
        "KEY": api_key,
        "Type": "xml",
        "pIndex": 1,
        "pSize": 100,
        "STATBL_ID": statbl_id,
        "DTACYCLE_CD": dtacycle_cd,
    }
    if start_wrttime:
        params["START_WRTTIME"] = start_wrttime
    if end_wrttime:
        params["END_WRTTIME"] = end_wrttime

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"API 호출 오류: {e}"

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        return f"XML 파싱 오류: {e}\n응답 내용: {resp.text[:300]}"

    # Check API-level error code
    code = root.findtext(".//CODE", root.findtext(".//code", ""))
    if code and code not in ("000", "INFO-000"):
        msg = root.findtext(".//MESSAGE", root.findtext(".//message", ""))
        return f"API 오류 (코드 {code}): {msg}"

    rows = root.findall(".//row")
    if not rows:
        return "조회된 통계 데이터가 없습니다."

    lines = [
        f"[한국부동산원 통계] STATBL_ID={statbl_id} / 주기={dtacycle_cd}"
        + (f" / {start_wrttime}~{end_wrttime}" if start_wrttime else "")
        + f" / {len(rows)}건\n"
    ]
    for r in rows:
        val = r.findtext("DTA_VAL", "").strip()
        lines.append(
            f"- [{r.findtext('WRTTIME_DESC','').strip()}] "
            f"{r.findtext('CLS_FULLNM', r.findtext('CLS_NM',''))} "
            f"/ {r.findtext('ITM_NM','').strip()} "
            f": {val} {r.findtext('UI_NM','').strip()}"
        )
    return "\n".join(lines)
