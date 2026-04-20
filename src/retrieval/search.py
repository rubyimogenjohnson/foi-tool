"""
search.py — semantic search over the pgvector FOI store.

Fetches top_k * 3 raw chunks via HNSW cosine distance, then deduplicates
by FOI identifier so we return one result per unique FOI (the
highest-scoring chunk from each).

hnsw.ef_search=100 is set per-connection to improve recall over the
default of 40, with negligible latency cost at this dataset size.
"""

from __future__ import annotations

from datetime import date, datetime

from src.db import get_conn
from src.ingest.embed import embed

_RECENCY_WINDOW_DAYS = 1095  # 3 years — FOIs outside this window score 0 on recency


def _parse_date(date_str: str) -> date | None:
    try:
        return datetime.strptime(date_str.strip(), "%d/%m/%Y").date()
    except (ValueError, AttributeError):
        return None


def _recency_score(date_str: str) -> float:
    """0–1 score: 1.0 = today, 0.0 = >= 3 years ago."""
    d = _parse_date(date_str)
    if d is None:
        return 0.0
    days_ago = (date.today() - d).days
    return max(0.0, 1.0 - days_ago / _RECENCY_WINDOW_DAYS)


def search(query: str, top_k: int = 5, recency_boost: float = 0.0) -> list[dict]:
    """
    Embed *query*, retrieve matching chunks, and return one result per FOI.

    Each result dict contains:
        identifier  — CAM reference (e.g. "CAM10600")
        title       — short subject line
        date        — document date string
        link        — URL to the full response / attachments zip
        excerpt     — the best-matching chunk text (title prepended)
        score       — relevance score 0–1 (higher is better)

    recency_boost: weight (0–1) given to date recency vs. semantic similarity.
        0.0  — pure semantic search (default, used by public portal)
        0.35 — blended ranking used by staff portal to surface latest FOIs
    """
    conn = get_conn()

    # With a recency boost we need more candidates before re-ranking
    n_over_fetch = top_k * 8 if recency_boost > 0 else top_k * 3

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM foi_chunks")
            total = cur.fetchone()[0]
            if total == 0:
                return []

        query_embedding = embed([query])[0]

        # Raise ef_search for this connection to improve recall.
        # HNSW default is 40; 100 gives better accuracy at negligible cost.
        with conn.cursor() as cur:
            cur.execute("SET hnsw.ef_search = 100")

            n_fetch = min(n_over_fetch, total)

            cur.execute(
                """
                SELECT
                    identifier,
                    title,
                    date,
                    link,
                    document,
                    1 - (embedding <=> %s::vector) AS score
                FROM foi_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, n_fetch),
            )
            rows = cur.fetchall()

    finally:
        conn.close()

    # Deduplicate by identifier, keeping the highest semantic score per FOI
    seen: dict[str, dict] = {}
    for identifier, title, date_str, link, document, score in rows:
        sem_score = float(score)
        if identifier not in seen or sem_score > seen[identifier]["_sem"]:
            seen[identifier] = {
                "identifier": identifier,
                "title": title,
                "date": date_str,
                "link": link,
                "excerpt": document,
                "_sem": sem_score,
            }

    candidates = list(seen.values())

    if recency_boost > 0:
        sem_weight = 1.0 - recency_boost
        for c in candidates:
            c["score"] = round(
                sem_weight * c["_sem"] + recency_boost * _recency_score(c["date"]),
                4,
            )
        candidates.sort(key=lambda c: c["score"], reverse=True)
    else:
        for c in candidates:
            c["score"] = round(c["_sem"], 4)
        candidates.sort(key=lambda c: c["score"], reverse=True)

    # Strip the internal key before returning
    for c in candidates:
        del c["_sem"]

    return candidates[:top_k]
