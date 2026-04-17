# FOI Tool — Architecture

## Overview

The tool has two distinct flows: an **ingest pipeline** (run once to populate the vector store) and a **query flow** (run on every user request via the Streamlit app).

---

## Ingest Pipeline

```mermaid
flowchart TD
    CSV["📄 Camden FOI CSV\ndata/*.csv"]

    subgraph ingest["Ingest Pipeline (src/ingest/)"]
        LOAD["Load rows\npipeline.py"]
        CLEAN["clean.py\nStrip header / address block / footer\nFix unicode encoding"]
        CHUNK["chunk.py\nSplit on paragraph boundaries\n800-char chunks, 150-char overlap"]
        EMBED["embed.py\nall-MiniLM-L6-v2\n384-dim embeddings\n(batches of 64)"]
        UPSERT["Upsert to ChromaDB\ncollection: foi_chunks\ncosine similarity space"]
    end

    CHROMA[("🗄️ ChromaDB\nchroma_db/")]

    CSV --> LOAD --> CLEAN --> CHUNK --> EMBED --> UPSERT --> CHROMA
```

Each chunk is stored with metadata: `identifier`, `title`, `date`, `link`, `chunk_index`, `total_chunks`. The pipeline is idempotent — re-running overwrites existing chunks by ID.

---

## Query Flow

```mermaid
flowchart TD
    USER["👤 User\nenters a question"]

    subgraph app["Streamlit App (src/app/main.py)"]
        INPUT["Text input"]
    end

    subgraph retrieval["Retrieval (src/retrieval/)"]
        SEMBED["embed.py\nEmbed query\nall-MiniLM-L6-v2"]
        SEARCH["search.py\nFetch top_k × 3 chunks\nfrom ChromaDB\nDeduplicate by FOI identifier\n→ top 5 unique FOIs"]
        FORMAT_CTX["format.py\nformat_context()\nNumbered blocks for LLM prompt"]
        FORMAT_SRC["format.py\nformat_sources()\nSource cards for UI"]
    end

    CHROMA[("🗄️ ChromaDB\nchroma_db/")]
    LLM["🤖 Claude Haiku\nAnswers using only\nprovided FOI excerpts\nwith CAM reference citations"]

    subgraph ui["Streamlit UI"]
        ANSWER["Answer text"]
        SOURCES["Expandable source cards\n(ref, title, date, relevance score, link)"]
    end

    USER --> INPUT --> SEMBED --> SEARCH
    CHROMA --> SEARCH
    SEARCH --> FORMAT_CTX --> LLM --> ANSWER
    SEARCH --> FORMAT_SRC --> SOURCES
```

---

## Component Summary

| Module | Responsibility |
|---|---|
| `src/ingest/clean.py` | Remove FOI boilerplate (header, address block, footer); fix encoding |
| `src/ingest/chunk.py` | Paragraph-aware splitting into 800-char overlapping chunks |
| `src/ingest/embed.py` | Shared `all-MiniLM-L6-v2` model; returns 384-dim float vectors |
| `src/ingest/pipeline.py` | Orchestrates full ingest: load → clean → chunk → embed → upsert |
| `src/retrieval/search.py` | Query embedding → ChromaDB → deduplicate to one result per FOI |
| `src/retrieval/format.py` | Shape hits into LLM context string and UI source dicts |
| `src/app/main.py` | Streamlit UI; calls retrieval then Claude Haiku for the final answer |
