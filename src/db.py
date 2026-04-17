"""
db.py — PostgreSQL connection and schema setup for pgvector.

Creates the foi_chunks table and an HNSW index for cosine similarity search.

HNSW was chosen over IVFFlat for this dataset (~12k FOIs / ~50-80k chunks) because:
  - Better recall without manual probes tuning
  - No training step required — index builds incrementally, safe for re-ingests
  - At this scale the extra memory vs IVFFlat is negligible

Index parameters:
  m=16             — connections per layer (default; higher = better recall, more memory)
  ef_construction=64 — candidate pool during build (default; higher = better quality, slower build)

Query parameter (set per-connection in search.py):
  hnsw.ef_search=100 — candidate pool at query time (default is 40; raising to 100 gives
                        better recall with negligible latency cost at this dataset size)
"""

from __future__ import annotations

import os

import psycopg2
from pgvector.psycopg2 import register_vector

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


def get_conn() -> psycopg2.extensions.connection:
    """Open a psycopg2 connection with pgvector types registered."""
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn


def ensure_schema(conn: psycopg2.extensions.connection) -> None:
    """
    Idempotently create the vector extension, table, and HNSW index.
    Safe to call on every ingest run — all statements use IF NOT EXISTS.
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS foi_chunks (
                id           TEXT PRIMARY KEY,
                identifier   TEXT NOT NULL,
                title        TEXT,
                date         TEXT,
                link         TEXT,
                chunk_index  INTEGER,
                total_chunks INTEGER,
                document     TEXT,
                embedding    vector(%s)
            )
        """, (EMBEDDING_DIM,))

        # HNSW index using cosine distance operator class.
        # Built incrementally — no need to drop/recreate on re-ingest.
        cur.execute("""
            CREATE INDEX IF NOT EXISTS foi_chunks_embedding_hnsw_idx
            ON foi_chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)

    conn.commit()
