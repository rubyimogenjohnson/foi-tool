"""
pipeline.py — ingest the Camden FOI CSV into a pgvector store.

Run from the project root:
    python -m src.ingest.pipeline
    python -m src.ingest.pipeline data/my_file.csv

The script:
  1. Loads all rows from the CSV
  2. Cleans each Document Text (strips boilerplate, fixes encoding)
  3. Chunks the cleaned text into ~800-char overlapping segments
  4. Prepends the FOI title to each chunk (so every chunk is self-describing)
  5. Embeds in batches of 64 using sentence-transformers
  6. Upserts into PostgreSQL (foi_chunks table) — safe to re-run, duplicates are overwritten
"""

from __future__ import annotations

import csv
import sys
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
from psycopg2.extras import execute_values

from src.db import ensure_schema, get_conn

load_dotenv()
from .clean import clean
from .chunk import chunk_text
from .embed import embed, get_model

DATA_DIR = Path(__file__).parents[2] / "data"
BATCH_SIZE = 64
YEARS = 3  # only ingest FOIs from the last N years


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_date(date_str: str) -> date | None:
    try:
        return datetime.strptime(date_str.strip(), "%d/%m/%Y").date()
    except ValueError:
        return None


def run(csv_path: Path | None = None) -> None:
    if csv_path is None:
        candidates = sorted(DATA_DIR.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
        csv_path = candidates[-1]

    records = _load_csv(csv_path)
    print(f"Loaded {len(records)} FOI records from {csv_path.name}")

    cutoff = date.today().replace(year=date.today().year - YEARS)
    records = [
        r for r in records
        if (d := _parse_date(r.get("Document Date", ""))) and d >= cutoff
    ]
    print(f"Filtered to {len(records)} records from the last {YEARS} years (since {cutoff})")

    print("Loading embedding model…")
    get_model()

    conn = get_conn(register=False)
    ensure_schema(conn)
    conn.close()

    all_rows: list[tuple] = []
    skipped = 0

    for record in records:
        identifier = record["Identifier"].strip()
        title = record.get("Document Title", "").strip()
        date_str = record.get("Document Date", "").strip()
        link = record.get("Document Link", "").strip()
        raw_text = record.get("Document Text", "")

        cleaned = clean(raw_text)
        if not cleaned:
            skipped += 1
            continue

        chunks = chunk_text(cleaned)
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            doc_text = f"{title}\n\n{chunk}" if title else chunk
            all_rows.append((
                f"{identifier}_{i}",  # id
                identifier,
                title,
                date_str,
                link,
                i,             # chunk_index
                total_chunks,
                doc_text,
            ))

    if skipped:
        print(f"Skipped {skipped} records with empty text after cleaning.")
    print(f"Embedding {len(all_rows)} chunks from {len(records) - skipped} records…")

    for start in range(0, len(all_rows), BATCH_SIZE):
        batch = all_rows[start:start + BATCH_SIZE]
        texts = [row[7] for row in batch]  # document column
        embeddings = embed(texts)

        rows_with_embeddings = [
            (*row, embedding)
            for row, embedding in zip(batch, embeddings)
        ]

        # Fresh connection per batch — avoids Supabase pooler timeouts
        conn = get_conn()
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO foi_chunks
                    (id, identifier, title, date, link,
                     chunk_index, total_chunks, document, embedding)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    identifier   = EXCLUDED.identifier,
                    title        = EXCLUDED.title,
                    date         = EXCLUDED.date,
                    link         = EXCLUDED.link,
                    chunk_index  = EXCLUDED.chunk_index,
                    total_chunks = EXCLUDED.total_chunks,
                    document     = EXCLUDED.document,
                    embedding    = EXCLUDED.embedding
                """,
                rows_with_embeddings,
            )
        conn.commit()
        conn.close()
        print(f"  {min(start + BATCH_SIZE, len(all_rows))}/{len(all_rows)} chunks upserted")

    print("\nDone. Vector store written to PostgreSQL (foi_chunks).")


if __name__ == "__main__":
    csv_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run(csv_arg)
