"""
cli.py — command-line interface for the Camden FOI RAG tool.

Usage:
    poetry run python -m src.app.cli
    poetry run python -m src.app.cli "how many looked-after children does Camden have?"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from dotenv import load_dotenv
load_dotenv()

import anthropic
from src.retrieval.search import search
from src.retrieval.format import format_context, format_sources

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


def ask(query: str) -> None:
    print(f"\nSearching FOI records for: {query!r}\n")
    hits = search(query, top_k=5)

    if not hits:
        print("No relevant FOI responses found.")
        return

    context = format_context(hits)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=_SYSTEM,
        messages=[{"role": "user", "content": f"Question: {query}\n\nRelevant FOI excerpts:\n\n{context}"}],
    )

    print("Answer:")
    print(message.content[0].text)
    print("\nSources:")
    for s in format_sources(hits):
        print(f"  [{s['ref']}] {s['title']} ({s['date']}) — score: {s['score']:.0%}")
        print(f"  {s['link']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        print("Camden FOI Search (type 'quit' to exit)\n")
        while True:
            try:
                query = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if query.lower() in ("quit", "exit", "q"):
                break
            if query:
                ask(query)
