"""
main.py — Streamlit front-end for the Camden FOI RAG tool.

Run from the project root:
    streamlit run src/app/main.py

Requires ANTHROPIC_API_KEY set in .env (loaded automatically by python-dotenv).
Requires the vector store to be populated first:
    python -m src.ingest.pipeline
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make absolute imports work when Streamlit runs this file as a script
sys.path.insert(0, str(Path(__file__).parents[2]))

import anthropic
import streamlit as st
from dotenv import load_dotenv

from src.retrieval.format import format_context, format_sources
from src.retrieval.search import search

load_dotenv()

TOP_K = 5

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


def _generate_answer(query: str, context: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Relevant FOI excerpts:\n\n{context}"
                ),
            }
        ],
    )
    return message.content[0].text


def main() -> None:
    st.set_page_config(page_title="Camden FOI Search", layout="centered")
    st.title("Camden FOI Search")
    st.caption(
        "Find out if your question has already been answered in a previous "
        "Freedom of Information response from Camden Council."
    )

    query = st.text_input(
        "What would you like to know?",
        placeholder="e.g. How many looked-after children does Camden have?",
    )

    if st.button("Search", type="primary", disabled=not query):
        with st.spinner("Searching FOI records…"):
            hits = search(query, top_k=TOP_K)

        if not hits:
            st.warning(
                "No relevant FOI responses found. "
                "You may want to submit a new FOI request."
            )
            return

        with st.spinner("Generating answer…"):
            context = format_context(hits)
            answer = _generate_answer(query, context)

        st.subheader("Answer")
        st.write(answer)

        st.divider()
        st.subheader("Source FOIs")
        st.caption("Expand each result to see relevance score and link to the full response.")

        for source in format_sources(hits):
            with st.expander(f"{source['ref']} — {source['title']} ({source['date']})"):
                st.progress(source["score"], text=f"Relevance: {source['score']:.0%}")
                st.markdown(f"[View full response & attachments]({source['link']})")


if __name__ == "__main__":
    main()
