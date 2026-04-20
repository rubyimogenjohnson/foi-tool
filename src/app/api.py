"""
api.py — FastAPI wrapper for the Camden FOI RAG tool.

Run from the project root:
    poetry run uvicorn src.app.api:app --reload

Endpoints:
    GET  /health       — liveness check
    POST /ask          — search FOIs and return an LLM-generated answer
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from dotenv import load_dotenv
load_dotenv()

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.ingest.embed import get_model
from src.retrieval.search import search
from src.retrieval.format import format_context, format_sources

app = FastAPI(title="Camden FOI Search API")

_SYSTEM = """\
You are a helpful assistant for Camden Council's Freedom of Information (FOI) service.
Your job is to help members of the public find out whether their question has already
been answered in a previous FOI response.

Rules:
- Answer using ONLY the provided FOI excerpts — never guess or add outside knowledge
- Always cite the FOI reference number (e.g. CAM10600) for any specific claim
- If the excerpts only partially answer the question, say so clearly
- If nothing in the excerpts is relevant, say the question does not appear to have
  been answered before and the person may wish to submit a new FOI request
- Write in plain, accessible English — avoid jargon
- Keep your answer concise (3–5 sentences is usually enough)
"""


@app.on_event("startup")
def _load_embedding_model() -> None:
    """Pre-load the sentence-transformer model so the first request isn't slow."""
    get_model()


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


class Source(BaseModel):
    ref: str
    title: str
    date: str
    link: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    hits = search(body.query, top_k=body.top_k)
    if not hits:
        return AskResponse(
            answer=(
                "No relevant FOI responses were found for your question. "
                "You may wish to submit a new FOI request to Camden Council."
            ),
            sources=[],
        )

    context = format_context(hits)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Question: {body.query}\n\nRelevant FOI excerpts:\n\n{context}",
            }
        ],
    )

    return AskResponse(
        answer=message.content[0].text,
        sources=[
            Source(
                ref=s["ref"],
                title=s["title"],
                date=s["date"],
                link=s["link"],
                score=s["score"],
            )
            for s in format_sources(hits)
        ],
    )
