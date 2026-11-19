"""
run_eval.py — Week-04 세 RAG 구성 평가 파이프라인

구성:
  baseline  — week-02 medical_aid + dense Top-5
  chunking  — Q&A 경계 청킹(Config A) + dense Top-5
  hybrid    — BM25 + dense → RRF Top-5 (Config B)

출력 (results/{config}/):
  raw_results.json   — 질문별 검색·생성·비용 기록
  eval_results.json  — Ragas 지표 + 사람 판정 필드

출력 (results/):
  comparison.json    — 질문별 세 구성 나란히 비교

CLI:
  python run_eval.py --dry-run   검색·Evidence Recall만 (API 호출 없음)
  python run_eval.py             전체 실행 (생성 + Ragas)

주의:
  - 판정 모델과 생성 모델이 동일(claude-sonnet-4-6) → self-judging bias 위험 있음
  - 비용 상수(INPUT_COST_PER_1M 등)는 실행 시점 Anthropic 가격표와 맞는지 확인
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
from typing import Optional

# ── 경로 추가 ─────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # rag_configs.py (같은 디렉토리)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "week-02", "proLmpa"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "week-03", "proLmpa"))

from dotenv import load_dotenv
load_dotenv()

from rag_basic import (
    retrieve,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)
from rag_configs import (
    build_qa_index,
    get_qa_collection,
    retrieve_qa,
    HybridRetriever,
    QA_COLLECTION_NAME,
)
import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

# ── 설정 ──────────────────────────────────────────────────────
EVAL_TOP_K = 5
GENERATION_MODEL = "claude-sonnet-4-6"
RESULTS_DIR = Path(_HERE) / "results"
QUESTIONS_FILE = Path(_HERE) / "eval_questions.json"

# Claude Sonnet 4.6 가격 (2024년 기준 — 실행 전 Anthropic 가격표 확인)
INPUT_COST_PER_1M = 3.0    # USD / 1M input tokens
OUTPUT_COST_PER_1M = 15.0  # USD / 1M output tokens

# 거절 판정 키워드 (week-02 원본 프롬프트 기준)
REFUSAL_PHRASES = [
    "확인할 수 없",
    "찾을 수 없",
    "해당 내용이 없",
    "제공된 문서에서",
]

# week-02 원본 시스템 프롬프트 (week-03 strict 프롬프트 사용 안 함 — 교란 방지)
SYSTEM_PROMPT = (
    "당신은 의료급여 제도 전문 안내원입니다. "
    "반드시 아래 제공된 문서 내용만을 근거로 답변하세요. "
    "문서에 관련 내용이 없으면 '제공된 문서에서 해당 내용을 찾을 수 없습니다.'라고 답하세요."
)


# ── 헬퍼 함수 ─────────────────────────────────────────────────

def build_context_string(hits: list[dict]) -> str:
    """generate()에 전달되는 컨텍스트 문자열 생성 (기록용)."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        blocks.append(
            f"[문서 {i}] (출처: {meta['source']}, {meta['page']}페이지, 유사도: {hit['score']})\n{hit['text']}"
        )
    return "\n\n".join(blocks)


def generate(query: str, hits: list[dict], client: Anthropic) -> dict:
    """
    week-02 원본 프롬프트로 답변 생성.
    Returns: {text, input_tokens, output_tokens, estimated_cost_usd}
    """
    context = build_context_string(hits)
    t0 = time.time()
    response = client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"{context}\n\n질문: {query}"}],
    )
    gen_ms = round((time.time() - t0) * 1000)
    inp = response.usage.input_tokens
    out = response.usage.output_tokens
    cost = (inp / 1_000_000 * INPUT_COST_PER_1M) + (out / 1_000_000 * OUTPUT_COST_PER_1M)
    return {
        "text": response.content[0].text,
        "generation_ms": gen_ms,
        "input_tokens": inp,
        "output_tokens": out,
        "estimated_cost_usd": round(cost, 6),
    }


def calc_evidence_recall(context_string: str, evidence_spans: list[str]) -> float | None:
    """Evidence Recall@5: evidence_spans 중 Top-5 컨텍스트에 포함된 비율."""
    if not evidence_spans:
        return None
    normalized = " ".join(context_string.split())
    found = sum(
        1 for span in evidence_spans
        if " ".join(span.split()) in normalized
    )
    return round(found / len(evidence_spans), 4)


def calc_page_recall(hits: list[dict], ground_truth_pages: list[int]) -> float | None:
    """Page Hit@5: 정답 페이지 중 Top-5에 포함된 비율."""
    if not ground_truth_pages:
        return None
    retrieved_pages = {h["metadata"].get("page") for h in hits}
    gt_pages = set(ground_truth_pages)
    return round(len(retrieved_pages & gt_pages) / len(gt_pages), 4)


def calc_answerability(answer: str, expected_answerable) -> dict:
    """거절 여부 키워드 판정."""
    refused = any(phrase in answer for phrase in REFUSAL_PHRASES)
    if expected_answerable is True:
        correct = not refused
    elif expected_answerable is False:
        correct = refused
    else:
        correct = None  # partial
    return {"refused": refused, "correct": correct}


def _hits_to_id_list(hits: list[dict], config_name: str) -> list[str]:
    """hits에서 청크 ID 문자열 목록 반환."""
    ids = []
    for h in hits:
        meta = h["metadata"]
        raw = meta.get("chunk_id")
        if config_name == "chunking":
            ids.append(f"qa_chunk_{raw}" if isinstance(raw, int) else str(raw))
        else:
            ids.append(f"chunk_{raw}" if isinstance(raw, int) else str(raw))
    return ids


# ── Ragas 평가 ────────────────────────────────────────────────

def run_ragas_evaluation(samples_data: list[dict], llm, embeddings) -> dict[str, dict]:
    """
    변경 사항 (week-03 대비):
      - FactualCorrectness(mode="recall"): 참조 사실 포함 여부 측정
      - ResponseRelevancy: unanswerable 질문(answerable=False) → N/A
      - Faithfulness: unanswerable 질문 → N/A (기존과 동일)
      - ContextRecall: evidence_spans 없는 질문 → N/A

    self-judging bias: 생성 모델과 판정 모델이 모두 claude-sonnet-4-6
    """
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import Faithfulness, FactualCorrectness, ResponseRelevancy, ContextRecall
    from ragas.dataset_schema import SingleTurnSample

    metric_results: dict[str, dict] = {}
    for s in samples_data:
        metric_results[s["id"]] = {
            "faithfulness": None,
            "faithfulness_na_reason": None,
            "factual_correctness_recall": None,
            "response_relevancy": None,
            "response_relevancy_na_reason": None,
            "context_recall": None,
            "context_recall_na_reason": None,
        }

    def _make_sample(s: dict) -> SingleTurnSample:
        return SingleTurnSample(
            user_input=s["user_input"],
            retrieved_contexts=s["retrieved_contexts"],
            response=s["response"],
            reference=s["reference"],
        )

    # ── FactualCorrectness(mode=recall) + ResponseRelevancy ──
    # ResponseRelevancy: unanswerable(answerable=False) 제외
    fc_samples, fc_ids = [], []
    rr_samples, rr_ids = [], []
    for s in samples_data:
        fc_samples.append(_make_sample(s))
        fc_ids.append(s["id"])
        if s.get("answerable") is not False:
            rr_samples.append(_make_sample(s))
            rr_ids.append(s["id"])
        else:
            metric_results[s["id"]]["response_relevancy_na_reason"] = (
                "거절 답변(unanswerable question)에는 관련성 점수가 무의미 → N/A"
            )

    if fc_samples:
        print("[Ragas] FactualCorrectness(mode=recall) 계산 중...")
        ds = EvaluationDataset(samples=fc_samples)
        result = evaluate(
            ds,
            metrics=[FactualCorrectness(mode="recall")],
            llm=llm,
            embeddings=embeddings,
        )
        df = result.to_pandas()
        # 컬럼명: "factual_correctness(mode=recall)" (Ragas 0.4.x 명명 규칙)
        fc_col = next(
            (c for c in df.columns if "factual_correctness" in c),
            None,
        )
        for i, qid in enumerate(fc_ids):
            if fc_col and i < len(df):
                v = df.iloc[i].get(fc_col)
                metric_results[qid]["factual_correctness_recall"] = (
                    None if (isinstance(v, float) and math.isnan(v)) else v
                )

    if rr_samples:
        print("[Ragas] ResponseRelevancy 계산 중...")
        ds = EvaluationDataset(samples=rr_samples)
        result = evaluate(ds, metrics=[ResponseRelevancy()], llm=llm, embeddings=embeddings)
        df = result.to_pandas()
        rr_col = next((c for c in df.columns if "answer_relevancy" in c or "response_relevancy" in c), None)
        for i, qid in enumerate(rr_ids):
            if rr_col and i < len(df):
                v = df.iloc[i].get(rr_col)
                metric_results[qid]["response_relevancy"] = (
                    None if (isinstance(v, float) and math.isnan(v)) else v
                )

    # ── Faithfulness: unanswerable 제외 ──
    faith_samples, faith_ids = [], []
    for s in samples_data:
        if s.get("answerable") is not False:
            faith_samples.append(_make_sample(s))
            faith_ids.append(s["id"])
        else:
            metric_results[s["id"]]["faithfulness_na_reason"] = (
                "거절 답변(unanswerable question)에는 검증할 주장이 없으므로 N/A"
            )

    if faith_samples:
        print("[Ragas] Faithfulness 계산 중...")
        ds = EvaluationDataset(samples=faith_samples)
        result = evaluate(ds, metrics=[Faithfulness()], llm=llm, embeddings=embeddings)
        df = result.to_pandas()
        for i, qid in enumerate(faith_ids):
            if i < len(df):
                v = df.iloc[i].get("faithfulness")
                metric_results[qid]["faithfulness"] = (
                    None if (isinstance(v, float) and math.isnan(v)) else v
                )

    # ── ContextRecall: evidence_spans 없는 질문 제외 ──
    cr_samples, cr_ids = [], []
    for s in samples_data:
        if s.get("has_evidence_spans") and s.get("answerable") is not False:
            cr_samples.append(_make_sample(s))
            cr_ids.append(s["id"])
        else:
            reason = (
                "unanswerable 질문 → N/A"
                if s.get("answerable") is False
                else "evidence_spans 없음 → N/A"
            )
            metric_results[s["id"]]["context_recall_na_reason"] = reason

    if cr_samples:
        print("[Ragas] ContextRecall 계산 중...")
        ds = EvaluationDataset(samples=cr_samples)
        result = evaluate(ds, metrics=[ContextRecall()], llm=llm, embeddings=embeddings)
        df = result.to_pandas()
        for i, qid in enumerate(cr_ids):
            if i < len(df):
                v = df.iloc[i].get("context_recall")
                metric_results[qid]["context_recall"] = (
                    None if (isinstance(v, float) and math.isnan(v)) else v
                )

    return metric_results


# ── 단일 구성 실행 ─────────────────────────────────────────────

def run_config(
    config_name: str,
    questions: list[dict],
    retriever_fn,          # (query, top_k) -> (hits, extra_stats)
    anthropic_client: Optional[Anthropic],
    dry_run: bool,
) -> list[dict]:
    """
    한 구성에 대해 모든 질문을 실행하고 raw_results 목록을 반환.
    dry_run=True: 검색·Evidence Recall만 (API 호출 없음)
    """
    raw_results = []
    print(f"\n{'=' * 70}")
    print(f"[{config_name.upper()}] {len(questions)}개 질문 실행 (top_k={EVAL_TOP_K})")
    print("=" * 70)

    for q in questions:
        qid = q["id"]
        query = q["question"]
        evidence_spans = q.get("evidence_spans", [])
        gt_pages = q.get("ground_truth_pages", [])
        print(f"\n[{qid}] {query[:60]}…")

        # 검색
        t_ret = time.time()
        result = retriever_fn(query, EVAL_TOP_K)
        retrieval_ms = round((time.time() - t_ret) * 1000)

        # retriever_fn 반환값 정규화: tuple(hits, stats) 또는 hits only
        if isinstance(result, tuple):
            hits, extra_stats = result
        else:
            hits, extra_stats = result, {}

        chunk_ids = _hits_to_id_list(hits, config_name)
        pages = [h["metadata"].get("page") for h in hits]
        scores = [h["score"] for h in hits]
        context_string = build_context_string(hits)

        print(f"  검색: {chunk_ids}  페이지: {pages}  ({retrieval_ms}ms)")

        # Evidence Recall
        ev_recall = calc_evidence_recall(context_string, evidence_spans)
        page_recall = calc_page_recall(hits, gt_pages)
        print(f"  Evidence Recall@5: {ev_recall}  Page Hit@5: {page_recall}")

        # 생성
        gen_info: dict = {}
        answer = ""
        if dry_run or anthropic_client is None:
            answer = "[dry-run: 생성 생략]"
            gen_info = {
                "generation_ms": 0,
                "input_tokens": None,
                "output_tokens": None,
                "estimated_cost_usd": None,
            }
        else:
            gen_info = generate(query, hits, anthropic_client)
            answer = gen_info.pop("text")
            print(f"  답변 길이: {len(answer)}자  생성: {gen_info['generation_ms']}ms  "
                  f"토큰: {gen_info['input_tokens']}in/{gen_info['output_tokens']}out  "
                  f"비용: ${gen_info['estimated_cost_usd']:.6f}")

        # Answerability
        answerability = calc_answerability(answer, q.get("answerable"))
        print(f"  거절: {answerability['refused']}  예상 일치: {answerability['correct']}")

        raw_results.append({
            "id": qid,
            "type": q.get("type"),
            "question": query,
            "config": config_name,
            "retrieved_chunk_ids": chunk_ids,
            "retrieved_pages": pages,
            "retrieved_scores": scores,
            "retrieved_texts": [h["text"] for h in hits],
            "context_string": context_string,
            "extra_retrieval_stats": extra_stats,
            "answer": answer,
            "evidence_spans": evidence_spans,
            "ground_truth_pages": gt_pages,
            "evidence_recall_at_5": ev_recall,
            "page_hit_at_5": page_recall,
            "answerability": answerability,
            "timing": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": gen_info.get("generation_ms", 0),
                "total_ms": retrieval_ms + gen_info.get("generation_ms", 0),
            },
            "cost": {
                "embedding_calls": 1,
                "bm25_calls": 1 if config_name == "hybrid" else 0,
                "llm_calls": 0 if dry_run else 1,
                "input_tokens": gen_info.get("input_tokens"),
                "output_tokens": gen_info.get("output_tokens"),
                "estimated_rag_cost_usd": gen_info.get("estimated_cost_usd"),
            },
            "generation_config": {
                "model": GENERATION_MODEL,
                "top_k": EVAL_TOP_K,
                "embedding_model": EMBEDDING_MODEL,
                "dry_run": dry_run,
            },
        })

    return raw_results


# ── 메인 ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Week-04 RAG 평가 파이프라인")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="검색·Evidence Recall만 실행 (LLM API 호출 없음)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=["baseline", "chunking", "hybrid"],
        default=["baseline", "chunking", "hybrid"],
        help="실행할 구성 선택 (기본: 전체)",
    )
    parser.add_argument(
        "--rebuild-chunking",
        action="store_true",
        help="Config A 컬렉션 강제 재빌드",
    )
    args = parser.parse_args()

    # 결과 디렉토리 생성
    for cfg in args.configs:
        (RESULTS_DIR / cfg).mkdir(parents=True, exist_ok=True)

    # 컴포넌트 초기화
    print(f"[초기화] SentenceTransformer: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    baseline_col = chroma_client.get_collection(COLLECTION_NAME)
    print(f"[초기화] baseline 컬렉션: {baseline_col.count()}개 청크")

    anthropic_client = None
    if not args.dry_run:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY 환경 변수가 필요합니다.")
        anthropic_client = Anthropic(api_key=api_key)

    # Config A 컬렉션
    qa_collection = None
    if "chunking" in args.configs:
        if args.rebuild_chunking:
            qa_collection = build_qa_index(embed_model)
        else:
            try:
                qa_collection = get_qa_collection()
                print(f"[Config A] 기존 컬렉션 사용: {qa_collection.count()}개 청크 (--rebuild-chunking 으로 재빌드)")
            except Exception:
                print("[Config A] 컬렉션 없음 → 빌드 시작")
                qa_collection = build_qa_index(embed_model)

    # Config B 하이브리드
    hybrid_retriever = None
    if "hybrid" in args.configs:
        hybrid_retriever = HybridRetriever(baseline_col, embed_model)

    # Ragas 초기화 (dry-run이 아닐 때만)
    ragas_llm = None
    ragas_embeddings = None
    if not args.dry_run:
        print("[Ragas] LLM 초기화: claude-sonnet-4-6")
        from langchain_anthropic import ChatAnthropic
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_core.embeddings import Embeddings as LCEmbeddings

        ragas_llm = LangchainLLMWrapper(
            ChatAnthropic(
                model=GENERATION_MODEL,
                api_key=os.environ["ANTHROPIC_API_KEY"],
            ),
            bypass_temperature=True,
        )

        class _STEmbeddings(LCEmbeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return embed_model.encode(texts, show_progress_bar=False).tolist()
            def embed_query(self, text: str) -> list[float]:
                return embed_model.encode([text], show_progress_bar=False).tolist()[0]

        ragas_embeddings = LangchainEmbeddingsWrapper(_STEmbeddings())

    # 질문 로드
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"\n[질문] {len(questions)}개 로드")

    # retriever_fn 맵핑 (모두 (query, top_k) -> (hits, stats) 반환)
    def _wrap_baseline(q, k):
        hits = retrieve(q, baseline_col, embed_model, top_k=k)
        return hits, {}

    def _wrap_chunking(q, k):
        hits = retrieve_qa(q, qa_collection, embed_model, top_k=k)
        return hits, {}

    def _wrap_hybrid(q, k):
        return hybrid_retriever.retrieve(q, top_k=k)

    retriever_map = {
        "baseline": _wrap_baseline,
        "chunking": _wrap_chunking,
        "hybrid": _wrap_hybrid,
    }

    all_raw: dict[str, list[dict]] = {}
    all_eval: dict[str, list[dict]] = {}

    # 각 구성 실행
    for config_name in args.configs:
        raw_results = run_config(
            config_name=config_name,
            questions=questions,
            retriever_fn=retriever_map[config_name],
            anthropic_client=anthropic_client,
            dry_run=args.dry_run,
        )
        all_raw[config_name] = raw_results

        # raw_results 저장
        raw_path = RESULTS_DIR / config_name / "raw_results.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, ensure_ascii=False, indent=2)
        print(f"\n[저장] {raw_path}")

        # Ragas 평가 (dry-run 아닐 때)
        if args.dry_run:
            eval_results = [{
                "id": r["id"],
                "type": r["type"],
                "question": r["question"],
                "config": config_name,
                "evidence_recall_at_5": r["evidence_recall_at_5"],
                "page_hit_at_5": r["page_hit_at_5"],
                "answerability_refused": r["answerability"]["refused"],
                "answerability_correct": r["answerability"]["correct"],
                "faithfulness": None,
                "factual_correctness_recall": None,
                "response_relevancy": None,
                "context_recall": None,
                "timing": r["timing"],
                "cost": r["cost"],
                "human_judgment": None,
                "human_notes": None,
            } for r in raw_results]
        else:
            # Ragas 입력 준비
            ragas_input = []
            for r in raw_results:
                q = next(qd for qd in questions if qd["id"] == r["id"])
                ragas_input.append({
                    "id": r["id"],
                    "user_input": r["question"],
                    "retrieved_contexts": r["retrieved_texts"],
                    "response": r["answer"],
                    "reference": q.get("reference", ""),
                    "answerable": q.get("answerable"),
                    "has_evidence_spans": bool(q.get("evidence_spans")),
                })

            print(f"\n[Ragas] {config_name} 평가 시작...")
            t_ragas = time.time()
            ragas_scores = run_ragas_evaluation(ragas_input, ragas_llm, ragas_embeddings)
            ragas_ms = round((time.time() - t_ragas) * 1000)
            print(f"[Ragas] 완료 ({ragas_ms}ms)")

            eval_results = []
            for r in raw_results:
                scores = ragas_scores.get(r["id"], {})
                eval_results.append({
                    "id": r["id"],
                    "type": r["type"],
                    "question": r["question"],
                    "config": config_name,
                    "evidence_recall_at_5": r["evidence_recall_at_5"],
                    "page_hit_at_5": r["page_hit_at_5"],
                    "answerability_refused": r["answerability"]["refused"],
                    "answerability_correct": r["answerability"]["correct"],
                    "faithfulness": scores.get("faithfulness"),
                    "faithfulness_na_reason": scores.get("faithfulness_na_reason"),
                    "factual_correctness_recall": scores.get("factual_correctness_recall"),
                    "response_relevancy": scores.get("response_relevancy"),
                    "response_relevancy_na_reason": scores.get("response_relevancy_na_reason"),
                    "context_recall": scores.get("context_recall"),
                    "context_recall_na_reason": scores.get("context_recall_na_reason"),
                    "timing": r["timing"],
                    "cost": r["cost"],
                    "eval_cost": {"ragas_time_ms": ragas_ms, "note": "Ragas LLM 호출 수 미추적; 시간만 기록"},
                    "human_judgment": None,   # pass / partial / fail
                    "human_notes": None,
                })

        eval_path = RESULTS_DIR / config_name / "eval_results.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        print(f"[저장] {eval_path}")
        all_eval[config_name] = eval_results

    # comparison.json 빌드
    comparison: dict[str, dict] = {}
    for config_name, eval_results in all_eval.items():
        for r in eval_results:
            qid = r["id"]
            if qid not in comparison:
                comparison[qid] = {}
            comparison[qid][config_name] = {
                "evidence_recall_at_5": r["evidence_recall_at_5"],
                "page_hit_at_5": r["page_hit_at_5"],
                "answerability_refused": r["answerability_refused"],
                "answerability_correct": r["answerability_correct"],
                "faithfulness": r.get("faithfulness"),
                "factual_correctness_recall": r.get("factual_correctness_recall"),
                "response_relevancy": r.get("response_relevancy"),
                "context_recall": r.get("context_recall"),
                "total_ms": r["timing"]["total_ms"],
                "estimated_rag_cost_usd": r["cost"].get("estimated_rag_cost_usd"),
                "human_judgment": r.get("human_judgment"),
            }

    comp_path = RESULTS_DIR / "comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {comp_path}")

    # 요약 출력
    _print_summary(all_eval, args.configs)


def _print_summary(all_eval: dict[str, list[dict]], configs: list[str]) -> None:
    print("\n" + "=" * 80)
    print("평가 결과 요약")
    print("=" * 80)

    def _fmt(v):
        if v is None:
            return "   N/A"
        if isinstance(v, bool):
            return "  True" if v else " False"
        return f"{v:>6.3f}"

    header = f"{'ID':<4} {'Config':<10} {'EvRecall':>8} {'Faith':>7} {'FC(rec)':>8} {'RespRel':>8} {'CtxRec':>7} {'ms':>6}"
    print(header)
    print("-" * 80)

    for qid in sorted({r["id"] for cfg_results in all_eval.values() for r in cfg_results}):
        for cfg in configs:
            if cfg not in all_eval:
                continue
            row = next((r for r in all_eval[cfg] if r["id"] == qid), None)
            if not row:
                continue
            print(
                f"{qid:<4} {cfg:<10} "
                f"{_fmt(row.get('evidence_recall_at_5')):>8} "
                f"{_fmt(row.get('faithfulness')):>7} "
                f"{_fmt(row.get('factual_correctness_recall')):>8} "
                f"{_fmt(row.get('response_relevancy')):>8} "
                f"{_fmt(row.get('context_recall')):>7} "
                f"{row['timing']['total_ms']:>6}"
            )
        print()

    print("=" * 80)
    print("\n다음 단계:")
    print("  1. 원본 PDF와 답변을 비교하여 각 results/*/eval_results.json 의 human_judgment 필드를 채우세요.")
    print("  2. eval_questions.json 의 evidence_spans 를 PDF 원문과 대조·수정하세요 (_note 참조).")
    print("  3. comparison.json 을 바탕으로 README.md 비교표를 작성하세요.")


if __name__ == "__main__":
    main()
