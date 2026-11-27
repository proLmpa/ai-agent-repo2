"""
run_scenarios.py
세 가지 요청 유형을 실행하고 각 실행의 Tool 호출 순서·입력·결과,
최종 응답, 종료 지점과 이유, 모델 호출 수, 실행 시간을 기록합니다.

실행: python run_scenarios.py
출력: 콘솔 출력 + run_log.md 저장
"""

import io
import json
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

# Windows 콘솔 cp949 인코딩 오류 방지
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from langchain_core.messages import AIMessage, ToolMessage

load_dotenv()

from agent import agent  # noqa: E402  (load_dotenv must run first)

TOOL_CALL_LIMIT = 5

SCENARIOS = [
    {
        "id": 1,
        "type": "유형 1 — 목표 달성 가능",
        "query": (
            "마포구 2024년 10월 아파트 매매 거래 사례를 찾아주고 "
            "최근 매매 시세 흐름도 알려줘."
        ),
    },
    {
        "id": 2,
        "type": "유형 2 — API로 확인 불가한 조건 포함",
        "query": (
            "초등학교 도보 5분 거리에 있고 조용한 마포구 아파트를 추천해줘."
        ),
    },
    {
        "id": 3,
        "type": "유형 3 — 앞선 실행과 다른 경로",
        "query": (
            "성동구에서 2024년 10월 기준으로 아파트 매매할지 전세로 살지 "
            "실거래 데이터를 비교해서 알려줘."
        ),
    },
]


# ---------------------------------------------------------------------------
# 실행 기록 파싱
# ---------------------------------------------------------------------------

def _extract_text(content) -> str:
    """AIMessage.content가 str 또는 list(content blocks)일 수 있어 통일합니다."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def parse_run(messages: list) -> dict:
    """메시지 리스트에서 Tool 호출 순서, 모델 호출 수, 최종 응답을 추출합니다."""

    # tool_call_id → 결과 텍스트 매핑 (ToolMessage 순회)
    result_map: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            result_map[msg.tool_call_id] = _extract_text(msg.content)

    tool_calls = []
    model_calls = 0

    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        model_calls += 1
        calls = getattr(msg, "tool_calls", None) or []
        for tc in calls:
            tool_calls.append(
                {
                    "name": tc["name"],
                    "args": tc["args"],
                    "result": result_map.get(tc["id"], "(결과 없음)"),
                }
            )

    # 최종 응답: tool_calls가 없는 마지막 AIMessage
    final_answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            calls = getattr(msg, "tool_calls", None) or []
            if not calls:
                final_answer = _extract_text(msg.content)
                break

    stop_reason = (
        f"반복 한도 초과 ({TOOL_CALL_LIMIT}회 제한)"
        if len(tool_calls) >= TOOL_CALL_LIMIT
        else "정상 완료 — 모델이 최종 응답 생성"
    )

    return {
        "tool_calls": tool_calls,
        "model_calls": model_calls,
        "final_answer": final_answer,
        "stop_reason": stop_reason,
    }


# ---------------------------------------------------------------------------
# 출력 포맷
# ---------------------------------------------------------------------------

def format_log(scenario: dict, run: dict, elapsed: float) -> str:
    SEP = "=" * 72
    lines = [
        SEP,
        f"시나리오 {scenario['id']}: {scenario['type']}",
        SEP,
        "",
        f"[사용자 요청]",
        scenario["query"],
        "",
        f"[Tool 호출 순서 — 총 {len(run['tool_calls'])}회]",
    ]

    if run["tool_calls"]:
        for i, tc in enumerate(run["tool_calls"], 1):
            args_str = json.dumps(tc["args"], ensure_ascii=False)
            result_preview = (
                tc["result"][:400] + " ..." if len(tc["result"]) > 400 else tc["result"]
            )
            lines += [
                f"",
                f"  ({i}) Tool: {tc['name']}",
                f"      입력: {args_str}",
                f"      결과: {result_preview}",
            ]
    else:
        lines.append("  (Tool 호출 없음)")

    lines += [
        "",
        "[최종 응답]",
        run["final_answer"],
        "",
        f"[종료 지점]  {run['stop_reason']}",
        f"[모델 호출 수]  {run['model_calls']}회",
        f"[실행 시간]  {elapsed:.2f}초",
        "",
        "[목표 달성 여부]  < 사람이 판정 후 기록 >",
        "[판정 근거]  < >",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = "\n".join([
        "# 실행 기록",
        "",
        f"- 실행 시각: {timestamp}",
        f"- 모델: claude-sonnet-4-6",
        f"- Tool 호출 한도: {TOOL_CALL_LIMIT}회",
        "",
    ])
    print(header)

    logs = [header]

    for scenario in SCENARIOS:
        print(f"\n{'─' * 40}")
        print(f"실행 중 → {scenario['type']}")
        print(f"요청: {scenario['query']}")
        print("─" * 40)

        start = time.time()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": scenario["query"]}]},
            config={"recursion_limit": 12},  # 최대 5회 Tool 호출 (node 2개 × 5 + 여유 2)
        )
        elapsed = time.time() - start

        run = parse_run(result["messages"])
        log = format_log(scenario, run, elapsed)

        print(log)
        logs.append(log)

    # run_log.md 저장
    output = "\n\n".join(logs)
    log_path = "run_log.md"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\n로그 저장 완료 → {log_path}")


if __name__ == "__main__":
    main()
