"""
rag_configs.py — Week-04 세 RAG 구성 정의

구성:
  Baseline   — week-02 medical_aid 컬렉션 + dense retrieve(top_k=5)
  Chunking   — Q&A 경계 기반 청킹 → medical_aid_qa 컬렉션 + dense retrieve(top_k=5)  [Config A]
  Hybrid     — Baseline 청크에 BM25(kiwipiepy) + dense Top-20 → RRF(k=60) → Top-5    [Config B]

CLI:
  python rag_configs.py --inspect
    → Config A: p.17 청크 출력 + 100자 미만 청크 없는지 확인
    → Config B: Q2 쿼리("틀니 1종 2종 본인부담률") BM25 Top-5 출력

사전 조건:
  - week-02 medical_aid ChromaDB 컬렉션 존재 (rag_basic.py 실행 후)
  - ANTHROPIC_API_KEY 불필요 (임베딩만 사용)
"""

import os
import re
import sys
import statistics
import argparse
from pathlib import Path

# ── week-02 경로 추가 ──────────────────────────────────────────
_W02 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "week-02", "proLmpa")
sys.path.insert(0, _W02)

from rag_basic import (
    retrieve,
    build_index,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PDF_PATH,
)
import chromadb
import pdfplumber
from sentence_transformers import SentenceTransformer

# ── 상수 ──────────────────────────────────────────────────────
QA_COLLECTION_NAME = "medical_aid_qa"
DATA_DIR = os.path.join(_W02, "data")

# Config A: 구조적 청킹 파라미터
QA_CHUNK_MIN_LEN = 100   # 이 미만은 노이즈로 버림
QA_CHUNK_MAX_LEN = 800   # 이 초과는 고정 크기로 폴백
QA_BOUNDARY_RE = re.compile(r"(?=Q\d+\.)")
# 페이지 헤더/풋터 패턴: "32 2024 알기 쉬운 의료급여제도 Ⅱ. 자주하는 질문 33"
NOISE_RE = re.compile(
    r"(?:\d{1,3}\s+)?2024 알기 쉬운 의료급여제도(?:\s+Ⅱ\.\s*자주하는 질문\s+\d{1,3})?",
)

# Config B: 하이브리드 파라미터
DENSE_TOP_N = 20
BM25_TOP_N = 20
RRF_K = 60

# ── 토크나이저 (kiwipiepy 없으면 공백 폴백) ─────────────────────
try:
    from kiwipiepy import Kiwi
    _kiwi = Kiwi()
    def _tokenize(text: str) -> list[str]:
        return [t.form for t in _kiwi.tokenize(text)]
    TOKENIZER_NAME = "kiwipiepy"
except ImportError:
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[\w가-힣]+", text)
    TOKENIZER_NAME = "whitespace_fallback"
    print("[경고] kiwipiepy 미설치 → 공백 기반 토크나이저 사용 (BM25 품질 저하 가능)")


# ══════════════════════════════════════════════════════════════
# 다중 PDF 로드 (테이블 + 텍스트)
# ══════════════════════════════════════════════════════════════

def load_pdf_with_tables(path: str) -> list[dict]:
    """
    pdfplumber로 텍스트와 구조화된 테이블을 함께 추출.
    반환: [{"page": int, "text": str, "tables": [[[cell, ...], ...]], "source": str}, ...]
    """
    source = os.path.basename(path)
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            if text.strip() or tables:
                pages.append({
                    "page": i,
                    "text": text.strip(),
                    "tables": tables,
                    "source": source,
                })
    return pages


def _format_table(table: list[list]) -> str:
    """테이블을 파이프 구분 텍스트로 변환. None 셀은 빈 문자열로."""
    rows = []
    for row in table:
        cells = [str(c).strip().replace("\n", " ") if c is not None else "" for c in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


# ══════════════════════════════════════════════════════════════
# Config A: 구조적 청킹 (테이블 + Q&A 경계 인식)
# ══════════════════════════════════════════════════════════════

def chunk_smart(pages: list[dict]) -> list[dict]:
    """
    테이블·Q&A 경계 인식 청킹. load_pdf_with_tables() 출력을 받는다.

    우선순위:
    1. 구조화된 테이블 → 각 테이블을 단일 청크 (800자 초과 시 10행 슬라이딩)
    2. Q&A 경계(Q\\d+.) 텍스트 분할
    3. 나머지 텍스트: 500/400 고정 크기 폴백

    source 필드가 있으면 메타데이터에 기록한다.
    """
    chunks = []
    chunk_id = 0

    for page_data in pages:
        page_num = page_data["page"]
        source = page_data.get("source", os.path.basename(PDF_PATH))
        text = page_data.get("text", "")
        tables = page_data.get("tables", [])

        # 1. 테이블 청크 ──────────────────────────────────────
        for tbl_idx, table in enumerate(tables):
            tbl_text = _format_table(table).strip()
            if len(tbl_text) < QA_CHUNK_MIN_LEN:
                continue
            if len(tbl_text) > QA_CHUNK_MAX_LEN:
                # 큰 테이블: 10행 슬라이딩, 2행 겹침
                rows = tbl_text.split("\n")
                step, overlap = 10, 2
                for start in range(0, len(rows), step - overlap):
                    sub = "\n".join(rows[start : start + step]).strip()
                    if len(sub) >= QA_CHUNK_MIN_LEN:
                        chunks.append({
                            "id": f"qa_chunk_{chunk_id}",
                            "text": sub,
                            "metadata": {
                                "source": source,
                                "page": page_num,
                                "chunk_id": chunk_id,
                                "method": "table_slide",
                                "table_index": tbl_idx,
                            },
                        })
                        chunk_id += 1
            else:
                chunks.append({
                    "id": f"qa_chunk_{chunk_id}",
                    "text": tbl_text,
                    "metadata": {
                        "source": source,
                        "page": page_num,
                        "chunk_id": chunk_id,
                        "method": "table",
                        "table_index": tbl_idx,
                    },
                })
                chunk_id += 1

        # 2. 텍스트: 헤더/풋터 제거 → Q&A 분할 → 고정 폴백 ──
        cleaned = NOISE_RE.sub("", text).strip()
        if not cleaned:
            continue

        parts = QA_BOUNDARY_RE.split(cleaned)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) > QA_CHUNK_MAX_LEN:
                start = 0
                while start < len(part):
                    sub = part[start : start + 500].strip()
                    if len(sub) >= QA_CHUNK_MIN_LEN:
                        chunks.append({
                            "id": f"qa_chunk_{chunk_id}",
                            "text": sub,
                            "metadata": {
                                "source": source,
                                "page": page_num,
                                "chunk_id": chunk_id,
                                "method": "fixed_fallback",
                            },
                        })
                        chunk_id += 1
                    start += 400
            elif len(part) >= QA_CHUNK_MIN_LEN:
                chunks.append({
                    "id": f"qa_chunk_{chunk_id}",
                    "text": part,
                    "metadata": {
                        "source": source,
                        "page": page_num,
                        "chunk_id": chunk_id,
                        "method": "qa_split",
                    },
                })
                chunk_id += 1

    return chunks


def chunk_qa(pages: list[dict]) -> list[dict]:
    """
    Q&A 경계(Q\\d+.)에서 분할. 100자 미만 청크(헤더·풋터 잔재) 제거.
    800자 초과 블록은 500/400 겹침 고정 크기 폴백.
    경계 패턴: Q\\d+.
    """
    chunks = []
    chunk_id = 0

    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]

        # 헤더·풋터 노이즈 제거
        cleaned = NOISE_RE.sub("", text).strip()
        if not cleaned:
            continue

        # Q&A 경계 분할
        parts = QA_BOUNDARY_RE.split(cleaned)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(part) > QA_CHUNK_MAX_LEN:
                # 고정 크기 폴백 (500자, 100자 겹침)
                start = 0
                while start < len(part):
                    sub = part[start : start + 500].strip()
                    if len(sub) >= QA_CHUNK_MIN_LEN:
                        chunks.append({
                            "id": f"qa_chunk_{chunk_id}",
                            "text": sub,
                            "metadata": {
                                "source": os.path.basename(PDF_PATH),
                                "page": page_num,
                                "chunk_id": chunk_id,
                                "method": "fixed_fallback",
                            },
                        })
                        chunk_id += 1
                    start += 400
            elif len(part) >= QA_CHUNK_MIN_LEN:
                chunks.append({
                    "id": f"qa_chunk_{chunk_id}",
                    "text": part,
                    "metadata": {
                        "source": os.path.basename(PDF_PATH),
                        "page": page_num,
                        "chunk_id": chunk_id,
                        "method": "qa_split",
                    },
                })
                chunk_id += 1
            # else: 100자 미만 → 버림

    return chunks


def build_qa_index(embed_model: SentenceTransformer) -> chromadb.Collection:
    """
    DATA_DIR 의 모든 PDF를 테이블+Q&A 청킹으로 로드해 medical_aid_qa 컬렉션 구축.
    기존 컬렉션은 삭제 후 재생성.
    """
    pdf_files = sorted(Path(DATA_DIR).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"PDF 파일 없음: {DATA_DIR}")

    all_pages: list[dict] = []
    for pdf_path in pdf_files:
        print(f"[Config A] 로드: {pdf_path.name}")
        pages = load_pdf_with_tables(str(pdf_path))
        all_pages.extend(pages)
        print(f"  → {len(pages)}페이지")

    chunks = chunk_smart(all_pages)

    lens = [len(c["text"]) for c in chunks]
    print(f"[Config A] 총 {len(chunks)}개 청크 ({len(pdf_files)}개 PDF)")
    print(f"  길이: 최소={min(lens)}, 최대={max(lens)}, "
          f"평균={statistics.mean(lens):.0f}, 중앙값={statistics.median(lens):.0f}")
    for method in ("table", "table_slide", "qa_split", "fixed_fallback"):
        n = sum(1 for c in chunks if c["metadata"].get("method") == method)
        if n:
            print(f"  {method}={n}개")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(QA_COLLECTION_NAME)
    except Exception:
        pass

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print("[Config A] 임베딩 중...")
    embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=False).tolist()

    collection = client.create_collection(
        QA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
    print(f"[Config A] 저장 완료 → {QA_COLLECTION_NAME} ({collection.count()}개 청크)")
    return collection


def get_qa_collection() -> chromadb.Collection:
    """기존 medical_aid_qa 컬렉션 반환. 없으면 예외."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(QA_COLLECTION_NAME)


def retrieve_qa(
    query: str,
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
    top_k: int = 5,
) -> list[dict]:
    """Config A 검색 (dense, QA 청킹 컬렉션). rag_basic.retrieve 재사용."""
    return retrieve(query, collection, embed_model, top_k=top_k)


# ══════════════════════════════════════════════════════════════
# Config B: 하이브리드 검색 (BM25 + Dense → RRF)
# ══════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    BM25(kiwipiepy 형태소) + Dense(cosine) → RRF(k=60) → Top-k.

    baseline medical_aid 컬렉션 위에서 동작한다.
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        embed_model: SentenceTransformer,
    ) -> None:
        from rank_bm25 import BM25Okapi

        self.collection = collection
        self.embed_model = embed_model

        # 전체 baseline 청크 로드
        result = collection.get(include=["documents", "metadatas"])
        self._ids: list[str] = result["ids"]
        self._docs: list[str] = result["documents"]
        self._metas: list[dict] = result["metadatas"]

        # BM25 인덱스 구축
        print(f"[Config B] BM25 인덱싱 (토크나이저: {TOKENIZER_NAME}, {len(self._docs)}개 문서)...")
        tokenized = [_tokenize(doc) for doc in self._docs]
        self.bm25 = BM25Okapi(tokenized)
        print("[Config B] BM25 준비 완료")

    # 내부 ID 형식 통일 (metadata chunk_id는 int)
    @staticmethod
    def _cid(meta: dict) -> str:
        raw = meta.get("chunk_id")
        return f"chunk_{raw}" if isinstance(raw, int) else str(raw)

    def retrieve(self, query: str, top_k: int = 5) -> tuple[list[dict], dict]:
        """
        Returns:
            hits      — top_k 청크 (text, metadata, score)
            stats     — dense_hits, bm25_hits, rrf_scores (디버그용)
        """
        # Dense Top-N
        qemb = self.embed_model.encode([query]).tolist()
        dense_result = self.collection.query(
            query_embeddings=qemb,
            n_results=DENSE_TOP_N,
            include=["documents", "metadatas", "distances"],
        )
        dense_hits: list[dict] = []
        for doc, meta, dist in zip(
            dense_result["documents"][0],
            dense_result["metadatas"][0],
            dense_result["distances"][0],
        ):
            dense_hits.append({
                "id": self._cid(meta),
                "text": doc,
                "metadata": meta,
                "score": round(1 - dist, 4),
            })

        # BM25 Top-N
        query_tokens = _tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_bm25_idx = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:BM25_TOP_N]
        bm25_hits: list[dict] = []
        for idx in top_bm25_idx:
            meta = self._metas[idx]
            bm25_hits.append({
                "id": self._cid(meta),
                "text": self._docs[idx],
                "metadata": meta,
                "score": round(float(bm25_scores[idx]), 4),
            })

        # RRF 병합
        rrf: dict[str, float] = {}
        for rank, h in enumerate(dense_hits):
            rrf[h["id"]] = rrf.get(h["id"], 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, h in enumerate(bm25_hits):
            rrf[h["id"]] = rrf.get(h["id"], 0.0) + 1.0 / (RRF_K + rank + 1)

        sorted_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:top_k]

        # 청크 데이터 복원 (dense 우선, 없으면 bm25)
        all_hits_map: dict[str, dict] = {}
        for h in bm25_hits + dense_hits:  # dense가 나중에 덮어씀
            all_hits_map[h["id"]] = h

        hits = [all_hits_map[cid] for cid in sorted_ids if cid in all_hits_map]
        stats = {
            "dense_top_ids": [h["id"] for h in dense_hits],
            "bm25_top_ids": [h["id"] for h in bm25_hits],
            "rrf_scores": {cid: round(score, 6) for cid, score in rrf.items()},
        }
        return hits, stats


# ══════════════════════════════════════════════════════════════
# --inspect: 구성 확인
# ══════════════════════════════════════════════════════════════

def _inspect(embed_model: SentenceTransformer) -> None:
    print("\n" + "=" * 70)
    print("▶ Config A: 다중 PDF + 테이블/Q&A 청킹 확인")
    print("=" * 70)

    # Build (or rebuild) QA index from all PDFs
    qa_collection = build_qa_index(embed_model)

    result = qa_collection.get(include=["documents", "metadatas"])
    items = sorted(
        zip(result["ids"], result["documents"], result["metadatas"]),
        key=lambda x: int(x[0].split("_")[-1]),
    )

    # PDF별 청크 수 집계
    from collections import Counter
    source_counts = Counter(meta.get("source", "unknown") for _, _, meta in items)
    method_counts = Counter(meta.get("method", "unknown") for _, _, meta in items)
    print(f"\nPDF별 청크 수:")
    for src, n in sorted(source_counts.items()):
        print(f"  {src}: {n}개")
    print(f"\n청킹 방법별:")
    for method, n in sorted(method_counts.items()):
        print(f"  {method}: {n}개")

    # 원본 PDF p.17 틀니 확인 (소스 필터)
    orig_src = os.path.basename(PDF_PATH)
    p17_chunks = [
        (cid, text, meta) for cid, text, meta in items
        if meta.get("page") == 17 and meta.get("source") == orig_src
    ]
    if p17_chunks:
        print(f"\n[{orig_src}] p.17 청크 {len(p17_chunks)}개:")
        for cid, text, meta in p17_chunks:
            print(f"  {cid} ({meta['method']}, {len(text)}자)")
            print(f"  미리보기: {text[:120].replace(chr(10), ' ')}…")
        a48_chunks = [cid for cid, text, _ in p17_chunks if "틀니" in text and "5%" in text]
        if a48_chunks:
            print(f"  [OK] A48(틀니 본인부담률) 포함 청크: {a48_chunks}")
        else:
            print("  [!!] A48 내용(틀니+5%)이 p.17 청크에서 발견되지 않음 — PDF 추출 확인 필요")
    else:
        print(f"  [{orig_src}] p.17 청크 없음")

    # 100자 미만 청크 확인
    short_chunks = [(cid, len(text)) for cid, text, _ in items if len(text) < QA_CHUNK_MIN_LEN]
    if short_chunks:
        print(f"\n  [!!] 100자 미만 청크 {len(short_chunks)}개 발견: {short_chunks}")
    else:
        print("  [OK] 100자 미만 청크 없음")

    print("\n" + "=" * 70)
    print("▶ Config B: Q2 쿼리 BM25 Top-5")
    print("=" * 70)

    # Load baseline collection
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    baseline_col = chroma_client.get_collection(COLLECTION_NAME)
    hybrid = HybridRetriever(baseline_col, embed_model)

    q2_query = "65세 이상 수급권자가 틀니 의료급여를 받을 때 1종과 2종의 본인부담률은 각각 얼마인가요?"
    hits, stats = hybrid.retrieve(q2_query, top_k=5)

    print(f"\nBM25 Top-5 (query: {q2_query[:40]}…):")
    for rank, cid in enumerate(stats["bm25_top_ids"][:5], 1):
        bm25_score = dict(zip(stats["bm25_top_ids"], [None] * BM25_TOP_N)).get(cid)
        print(f"  {rank}. {cid}")
    print("\nRRF Top-5:")
    for rank, hit in enumerate(hits, 1):
        print(f"  {rank}. {hit['id']} (p.{hit['metadata'].get('page')}) RRF={stats['rrf_scores'].get(hit['id'], '?'):.6f}")
        print(f"     미리보기: {hit['text'][:80].replace(chr(10), ' ')}…")

    if any(h["id"] == "chunk_59" for h in hits):
        print("\n  [OK] chunk_59 가 RRF Top-5에 포함됨")
    else:
        print("\n  [!!] chunk_59 미포함 — 하이브리드 검색도 실패 (Q2 여전히 어려움)")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 구성 검사 도구")
    parser.add_argument("--inspect", action="store_true", help="Config A/B 구성 확인")
    args = parser.parse_args()

    if args.inspect:
        from dotenv import load_dotenv
        load_dotenv()
        print(f"[초기화] SentenceTransformer: {EMBEDDING_MODEL}")
        embed_model = SentenceTransformer(EMBEDDING_MODEL)
        _inspect(embed_model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
