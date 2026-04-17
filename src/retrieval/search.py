"""
search.py — semantic search over the pgvector FOI store.

Fetches top_k * 3 raw chunks via HNSW cosine distance, then deduplicates
by FOI identifier so we return one result per unique FOI (the
highest-scoring chunk from each).

hnsw.ef_search=100 is set per-connection to improve recall over the
default of 40, with negligible latency cost at this dataset size.
"""

from __future__ import annotations

from src.db import get_conn
from src.ingest.embed import embed


def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed *query*, retrieve matching chunks, and return one result per FOI.

    Each result dict contains:
        identifier  — CAM reference (e.g. "CAM10600")
        title       — short subject line
        date        — document date string
        link        — URL to the full response / attachments zip
        excerpt     — the best-matching chunk text (title prepended)
        score       — cosine similarity 0–1 (higher is better)
    """
    conn = get_conn()

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

            # Over-fetch so deduplication has enough candidates
            n_fetch = min(top_k * 3, total)

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

    hits: list[dict] = []
    seen: set[str] = set()

    for identifier, title, date, link, document, score in rows:
        if identifier in seen:
            continue  # already have the best chunk for this FOI
        seen.add(identifier)

        hits.append({
            "identifier": identifier,
            "title": title,
            "date": date,
            "link": link,
            "excerpt": document,
            "score": round(float(score), 4),
        })

        if len(hits) >= top_k:
            break

    return hits
