"""
format.py — shape search hits into LLM context and UI source citations.
"""


def format_context(hits: list[dict]) -> str:
    """
    Format retrieved FOI chunks as numbered blocks for the LLM prompt.
    Each block is headed with the reference, title, and date so the model
    can cite them accurately.
    """
    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(
            f"[{i}] {hit['identifier']} — {hit['title']} ({hit['date']})\n"
            f"{hit['excerpt']}"
        )
    return "\n\n---\n\n".join(parts)


def format_sources(hits: list[dict]) -> list[dict]:
    """Return source metadata ready for display in the Streamlit UI."""
    return [
        {
            "ref": h["identifier"],
            "title": h["title"],
            "date": h["date"],
            "link": h["link"],
            "score": h["score"],
        }
        for h in hits
    ]
