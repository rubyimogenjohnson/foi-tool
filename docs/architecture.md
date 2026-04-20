# FOI Tool — Architecture

```mermaid
flowchart TD
    CSV["📄 Camden FOI CSV\ndata/*.csv"]

    subgraph ingest["Ingest pipeline — src/ingest/"]
        CLEAN["clean.py\nStrip boilerplate · fix encoding"]
        CHUNK["chunk.py\n800-char overlapping chunks\n150-char overlap"]
        EMBED_I["embed.py\nall-MiniLM-L6-v2\n384-dim vectors · batches of 64"]
        UPSERT["pipeline.py\nUpsert to PostgreSQL\nON CONFLICT DO UPDATE"]
    end

    DB[("🗄️ PostgreSQL + pgvector\nfoi_chunks — HNSW cosine index\nfoi_feedback — user votes")]

    subgraph flask["Flask app — src/app/flask_app.py"]

        subgraph public["Public portal  /"]
            PQ["User question"]
            PS["search.py\ntop 5 FOIs · no recency boost"]
            PL["Claude Haiku\npublic system prompt"]
            PA["Answer + source cards"]
        end

        subgraph staff["Staff portal  /staff"]
            SQ["Staff question"]
            SS["search.py\ntop 8 FOIs · +40%% recency boost"]
            SL["Claude Haiku\nstaff system prompt"]
            SA["Answer + sources table\n(tabs)"]
        end

        FB["Feedback widget\n👍 / 👎 stored in foi_feedback"]
        STATS["Feedback stats  /staff/stats"]
    end

    subgraph annotation["Annotation  /staff/eval/annotate"]
        RAND["Random FOI from DB"]
        HUMAN["Human reviewer\nwrites question"]
        EQ["eval_questions.json\nground-truth Q&A pairs"]
    end

    subgraph evaluation["Evaluation  /staff/eval"]
        SWEEP["run_eval_sweep()\nSearch at every K = 1…20"]
        EVAL_LLM["Claude Haiku\nper-question answer"]
        METRICS["Precision@K · Recall@K\nHit Rate · MRR · Mean Rank"]
        RUNS["eval_runs.json\nhistorical run log"]
        DASH["Eval dashboard\ninteractive charts + per-question table"]
    end

    %% Ingest flow
    CSV --> CLEAN --> CHUNK --> EMBED_I --> UPSERT --> DB

    %% Public portal flow
    PQ --> PS
    DB -->|"HNSW cosine search"| PS
    PS --> PL --> PA
    PA --> FB --> DB

    %% Staff portal flow
    SQ --> SS
    DB -->|"HNSW cosine search"| SS
    SS --> SL --> SA
    SA --> FB
    DB --> STATS

    %% Annotation flow
    DB -->|"ORDER BY RANDOM()"| RAND --> HUMAN --> EQ

    %% Evaluation flow
    EQ --> SWEEP
    DB -->|"HNSW cosine search"| SWEEP
    SWEEP --> METRICS
    SWEEP --> EVAL_LLM
    METRICS --> RUNS
    EVAL_LLM --> RUNS
    RUNS --> DASH
```

## Component summary

| Module | Responsibility |
|---|---|
| `src/ingest/clean.py` | Remove FOI boilerplate (header, address block, footer); fix encoding |
| `src/ingest/chunk.py` | Paragraph-aware splitting into 800-char overlapping chunks |
| `src/ingest/embed.py` | Shared `all-MiniLM-L6-v2` model; returns 384-dim float vectors |
| `src/ingest/pipeline.py` | Orchestrates full ingest: load → clean → chunk → embed → upsert |
| `src/db.py` | Connection helper; schema setup (foi_chunks HNSW index, foi_feedback table) |
| `src/retrieval/search.py` | Query embedding → pgvector HNSW → deduplicate to one result per FOI; optional recency boost |
| `src/retrieval/format.py` | Shape hits into LLM context string and UI source dicts |
| `src/app/flask_app.py` | All routes: public portal, staff portal, feedback, eval dashboard, annotation |
| `src/eval/run.py` | Evaluation logic: search sweep across K values; precision, recall, MRR, mean rank; LLM answer generation |
