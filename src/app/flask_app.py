"""
flask_app.py — Flask front-end for the Camden FOI RAG tool.

Two portals:
  /        — Public portal: members of the public search for similar past FOIs
  /staff   — Staff portal:  internal staff check what Camden has said before,
             with recency-weighted ranking to surface the most recent answers
  /staff/eval — Retrieval evaluation dashboard

Run from the project root:
    flask --app src.app.flask_app run --debug

Requires ANTHROPIC_API_KEY and DATABASE_URL set in .env.
Requires the vector store to be populated first:
    python -m src.ingest.pipeline
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import anthropic
import markdown as md
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for
from markupsafe import Markup

from src.db import ensure_feedback_table, ensure_query_log_table, get_conn
from src.eval.run import (
    aggregate_metrics,
    load_questions,
    run_eval,
    run_eval_sweep,
)
from src.retrieval.format import format_context, format_sources
from src.retrieval.search import search

EVAL_PATH     = Path(__file__).parents[2] / "data" / "eval_questions.json"
RUNS_FILE     = Path(__file__).parents[2] / "data" / "eval_runs.json"
DEFAULT_MAX_K = 20

load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="")


def _init_feedback_table() -> None:
    try:
        conn = get_conn(register=False)
        ensure_feedback_table(conn)
        ensure_query_log_table(conn)
        conn.close()
    except Exception:
        pass  # don't crash startup if DB is unavailable


def _log_query(portal: str, query: str, answer: str, n_sources: int) -> None:
    """Write a query + answer to foi_query_log. Silent on failure."""
    try:
        conn = get_conn(register=False)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO foi_query_log (portal, query, answer, n_sources) VALUES (%s, %s, %s, %s)",
                (portal, query, answer, n_sources),
            )
        conn.commit()
        conn.close()
    except Exception:
        pass


@app.template_filter("markdown")
def markdown_filter(text: str) -> Markup:
    """Convert markdown text to safe HTML for use in templates."""
    return Markup(md.markdown(text, extensions=["nl2br"]))

TOP_K_PUBLIC = 5
TOP_K_STAFF = 8
RECENCY_BOOST_PUBLIC = 0.25   # 25% recency, 75% semantic
RECENCY_BOOST_STAFF  = 0.40   # 40% recency, 60% semantic

_PUBLIC_SYSTEM = """\
You are a helpful assistant for Camden Council's Freedom of Information (FOI) service.
Your job is to help members of the public find out whether their question has already
been answered in a previous FOI response.

Rules:
- Answer using ONLY the provided FOI excerpts — never guess or add outside knowledge
- For every specific claim, cite the FOI reference number AND the date it was answered,
  e.g. "According to CAM10600 (answered 12 March 2024), ..."
- Prioritise the most recent responses — if an older and a newer response address the
  same point, favour the newer one and note that it is the latest available answer
- If the excerpts only partially answer the question, say so clearly
- If nothing in the excerpts is relevant, say the question does not appear to have
  been answered before and the person may wish to submit a new FOI request
- Write in plain, accessible English — avoid jargon
- Keep your answer concise (3–5 sentences is usually enough)
"""

_STAFF_SYSTEM = """\
You are an assistant for Camden Council's internal FOI team.
Your job is to help staff quickly understand what Camden has previously said on a topic
so they can respond consistently to new FOI requests.

Rules:
- For every point you make, cite the FOI reference number AND the date it was answered,
  e.g. "CAM10600 (12 March 2024) stated that..."
- Always lead with the most recent response on each point
- If responses on the same topic have changed over time, highlight the change and the dates
- If the excerpts only partially cover the topic, say so
- If nothing is relevant, say so clearly — do not invent information
- Be concise and factual; bullet points are preferred
"""


def _llm_answer(query: str, context: str, system: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system,
        messages=[
            {
                "role": "user",
                "content": f"Question: {query}\n\nRelevant FOI excerpts:\n\n{context}",
            }
        ],
    )
    return msg.content[0].text


def _days_ago_label(date_str: str) -> str:
    """Return a human-friendly recency label, e.g. '3 months ago'."""
    try:
        d = datetime.strptime(date_str.strip(), "%d/%m/%Y").date()
    except (ValueError, AttributeError):
        return ""
    days = (datetime.today().date() - d).days
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    if days < 31:
        weeks = days // 7
        return f"{weeks} week{'s' if weeks > 1 else ''} ago"
    if days < 365:
        months = days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    years = days // 365
    return f"{years} year{'s' if years > 1 else ''} ago"


# ---------------------------------------------------------------------------
# Public portal
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def public_index():
    query = ""
    answer = None
    sources = []
    error = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if not query:
            error = "Please enter a question to search."
        else:
            hits = search(query, top_k=TOP_K_PUBLIC, recency_boost=RECENCY_BOOST_PUBLIC)
            if not hits:
                error = (
                    "No relevant FOI responses were found for your question. "
                    "You may wish to submit a new FOI request to Camden Council."
                )
            else:
                context = format_context(hits)
                answer = _llm_answer(query, context, _PUBLIC_SYSTEM)
                sources = format_sources(hits)
                for s in sources:
                    s["days_ago"] = _days_ago_label(s["date"])
                _log_query("public", query, answer, len(hits))

    return render_template(
        "public.html",
        query=query,
        answer=answer,
        sources=sources,
        error=error,
    )


# ---------------------------------------------------------------------------
# Staff portal
# ---------------------------------------------------------------------------

@app.route("/staff", methods=["GET", "POST"])
def staff_index():
    query = ""
    answer = None
    sources = []
    error = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if not query:
            error = "Please enter a topic or question to search."
        else:
            hits = search(query, top_k=TOP_K_STAFF, recency_boost=RECENCY_BOOST_STAFF)
            if not hits:
                error = (
                    "No relevant previous FOI responses were found. "
                    "Camden may not have been asked about this topic before."
                )
            else:
                context = format_context(hits)
                answer = _llm_answer(query, context, _STAFF_SYSTEM)
                sources = format_sources(hits)
                for s in sources:
                    s["days_ago"] = _days_ago_label(s["date"])
                _log_query("staff", query, answer, len(hits))

    return render_template(
        "staff.html",
        query=query,
        answer=answer,
        sources=sources,
        error=error,
    )


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}
    vote = data.get("vote", "").strip().lower()
    portal = data.get("portal", "").strip().lower()
    query = data.get("query", "").strip()

    if vote not in ("yes", "no") or portal not in ("public", "staff"):
        return jsonify({"error": "invalid"}), 400

    try:
        conn = get_conn(register=False)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO foi_feedback (portal, query, vote) VALUES (%s, %s, %s)",
                (portal, query or None, vote),
            )
        conn.commit()
        conn.close()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"ok": True})


@app.route("/staff/logs")
def staff_logs():
    portal = request.args.get("portal", "all")
    page   = max(1, int(request.args.get("page", 1)))
    per_page = 25

    try:
        conn = get_conn(register=False)
        with conn.cursor() as cur:
            where = "" if portal == "all" else "WHERE portal = %s"
            params_count = () if portal == "all" else (portal,)
            cur.execute(f"SELECT COUNT(*) FROM foi_query_log {where}", params_count)
            total = cur.fetchone()[0]

            offset = (page - 1) * per_page
            params = () if portal == "all" else (portal,)
            cur.execute(
                f"""
                SELECT id, portal, query, answer, n_sources, created_at
                FROM foi_query_log
                {where}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (*params, per_page, offset),
            )
            rows = cur.fetchall()
        conn.close()
    except Exception as exc:
        return f"<p>Error: {exc}</p>", 500

    logs = [
        {
            "id":         r[0],
            "portal":     r[1],
            "query":      r[2],
            "answer":     r[3],
            "n_sources":  r[4],
            "created_at": r[5].strftime("%d %b %Y %H:%M") if r[5] else "",
        }
        for r in rows
    ]

    total_pages = max(1, -(-total // per_page))  # ceiling division

    return render_template(
        "logs.html",
        logs=logs,
        portal=portal,
        page=page,
        total=total,
        total_pages=total_pages,
        per_page=per_page,
    )


@app.route("/staff/stats")
def staff_stats():
    try:
        conn = get_conn(register=False)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    portal,
                    vote,
                    COUNT(*)                              AS total,
                    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '7 days')  AS last_7_days,
                    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '30 days') AS last_30_days
                FROM foi_feedback
                GROUP BY portal, vote
                ORDER BY portal, vote
            """)
            rows = cur.fetchall()
        conn.close()
    except Exception as exc:
        return f"<p>Error: {exc}</p>", 500

    # Shape into {portal: {yes: {...}, no: {...}}}
    stats: dict = {}
    for portal, vote, total, last7, last30 in rows:
        stats.setdefault(portal, {})[vote] = {
            "total": total,
            "last_7_days": last7,
            "last_30_days": last30,
        }

    return render_template("stats.html", stats=stats)


# ---------------------------------------------------------------------------
# Evaluation — annotation (human dataset builder)
# ---------------------------------------------------------------------------

def _get_random_foi() -> tuple[str, list[dict], dict | None]:
    """Pick a random FOI identifier and return its chunks."""
    conn = get_conn(register=False)
    try:
        with conn.cursor() as cur:
            # Pick one random identifier
            cur.execute("""
                SELECT identifier FROM foi_chunks
                ORDER BY RANDOM()
                LIMIT 1
            """)
            row = cur.fetchone()
            if not row:
                return "", [], None
            identifier = row[0]

            cur.execute("""
                SELECT chunk_index, total_chunks, document, title, date, link
                FROM foi_chunks
                WHERE identifier = %s
                ORDER BY chunk_index
            """, (identifier,))
            rows = cur.fetchall()
    finally:
        conn.close()

    chunks = [{"index": r[0], "total": r[1], "text": r[2]} for r in rows]
    meta   = {"title": rows[0][3], "date": rows[0][4], "link": rows[0][5]}
    return identifier, chunks, meta


@app.route("/staff/eval/annotate")
def eval_annotate():
    saved = request.args.get("saved", "")

    identifier, foi_chunks, foi_meta = _get_random_foi()

    existing_questions = json.loads(EVAL_PATH.read_text()) if EVAL_PATH.exists() else []
    already_covered = [
        q for q in existing_questions
        if identifier and identifier in q.get("relevant_identifiers", [])
    ]

    return render_template(
        "annotate.html",
        identifier=identifier,
        foi_chunks=foi_chunks,
        foi_meta=foi_meta,
        already_covered=already_covered,
        saved=saved,
    )


@app.route("/staff/eval/annotate/save", methods=["POST"])
def eval_annotate_save():
    identifier  = request.form.get("identifier", "").strip()
    question    = request.form.get("question", "").strip()
    identifiers = [i.strip() for i in request.form.get("identifiers", identifier).split(",") if i.strip()]

    if not identifiers or not question:
        return redirect(url_for("eval_annotate", identifier=identifier, error="1"))

    questions = json.loads(EVAL_PATH.read_text()) if EVAL_PATH.exists() else []

    # Auto-generate next ID
    existing_ids = [q.get("id", "") for q in questions]
    n = len(questions) + 1
    while f"q{n:02d}" in existing_ids:
        n += 1
    new_id = f"q{n:02d}"

    questions.append({
        "id":                   new_id,
        "question":             question,
        "relevant_identifiers": identifiers,
    })
    EVAL_PATH.write_text(json.dumps(questions, indent=2))

    return redirect(url_for("eval_annotate", identifier=identifier, saved=new_id))


# ---------------------------------------------------------------------------
# Evaluation dashboard
# ---------------------------------------------------------------------------

def _load_runs() -> list[dict]:
    if not RUNS_FILE.exists():
        return []
    return json.loads(RUNS_FILE.read_text())


def _save_run(run: dict) -> None:
    runs = _load_runs()
    runs.append(run)
    RUNS_FILE.write_text(json.dumps(runs, indent=2))


def _build_chart_data(sweep: dict[int, dict]) -> dict:
    k_sorted = sorted(sweep.keys())
    return {
        "k":         k_sorted,
        "precision": [sweep[k]["mean_precision"] for k in k_sorted],
        "recall":    [sweep[k]["mean_recall"]    for k in k_sorted],
        "hit_rate":  [sweep[k]["hit_rate"]       for k in k_sorted],
        "mrr":       [sweep[k]["mrr"]            for k in k_sorted],
    }


def _build_rank_dist(per_question: list[dict], max_k: int) -> dict:
    rank_counts: dict[str, int] = {}
    not_found = 0
    for r in per_question:
        if r["rank"] is None:
            not_found += 1
        else:
            key = str(r["rank"])
            rank_counts[key] = rank_counts.get(key, 0) + 1
    return {
        "labels": [str(r) for r in range(1, max_k + 1)] + ["not found"],
        "counts": [rank_counts.get(str(r), 0) for r in range(1, max_k + 1)] + [not_found],
    }


@app.route("/staff/eval")
def eval_dashboard():
    questions = load_questions(EVAL_PATH)
    runs = _load_runs()

    if not runs:
        return render_template(
            "eval.html",
            n_questions=len(questions),
            runs=[],
            selected=None,
            sweep_json=None,
            rank_dist_json=None,
            default_max_k=DEFAULT_MAX_K,
        )

    run_id   = request.args.get("run")
    selected = next((r for r in runs if r["id"] == run_id), None) or runs[-1]

    sweep      = {int(k): v for k, v in selected["sweep"].items()}
    max_k      = selected["max_k"]
    sweep_data = _build_chart_data(sweep)
    rank_dist  = _build_rank_dist(selected["per_question"], max_k)

    return render_template(
        "eval.html",
        n_questions=len(questions),
        runs=list(reversed(runs)),
        selected=selected,
        sweep_json=json.dumps(sweep_data),
        rank_dist_json=json.dumps(rank_dist),
        default_max_k=DEFAULT_MAX_K,
    )


@app.route("/staff/eval/run", methods=["POST"])
def eval_run():
    max_k = int(request.form.get("max_k", DEFAULT_MAX_K))
    max_k = max(1, min(max_k, 50))

    questions    = load_questions(EVAL_PATH)
    k_values     = list(range(1, max_k + 1))
    sweep        = run_eval_sweep(questions, k_values)
    per_question = run_eval(questions, top_k=max_k, generate_answers=True)
    metrics      = aggregate_metrics(per_question)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_run({
        "id":           run_id,
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "max_k":        max_k,
        "n_questions":  len(questions),
        "metrics":      metrics,
        "sweep":        {str(k): v for k, v in sweep.items()},
        "per_question": per_question,
    })
    return redirect(url_for("eval_dashboard", run=run_id))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

_init_feedback_table()

if __name__ == "__main__":
    app.run(debug=True)
