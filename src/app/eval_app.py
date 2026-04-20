"""
eval_app.py — standalone Flask app for retrieval evaluation.

Run from the project root:
    flask --app src.app.eval_app run --port 5001 --debug

Routes:
  GET  /            — redirect to /eval
  GET  /eval        — dashboard; ?run=<run_id> selects a historical run
  POST /eval/run    — runs a new evaluation (form field: max_k)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, url_for

load_dotenv()

from src.eval.run import (
    aggregate_metrics,
    load_questions,
    run_eval,
    run_eval_sweep,
)

EVAL_PATH  = Path(__file__).parents[2] / "data" / "eval_questions.json"
RUNS_FILE  = Path(__file__).parents[2] / "data" / "eval_runs.json"
DEFAULT_MAX_K = 20

app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="")


# ---------------------------------------------------------------------------
# Run storage
# ---------------------------------------------------------------------------

def _load_runs() -> list[dict]:
    if not RUNS_FILE.exists():
        return []
    return json.loads(RUNS_FILE.read_text())


def _save_run(run: dict) -> None:
    runs = _load_runs()
    runs.append(run)
    RUNS_FILE.write_text(json.dumps(runs, indent=2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return redirect(url_for("eval_dashboard"))


@app.route("/eval")
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

    # Pick which run to display — query param or most recent
    run_id   = request.args.get("run")
    selected = next((r for r in runs if r["id"] == run_id), None) or runs[-1]

    sweep       = {int(k): v for k, v in selected["sweep"].items()}
    max_k       = selected["max_k"]
    sweep_data  = _build_chart_data(sweep)
    rank_dist   = _build_rank_dist(selected["per_question"], max_k)

    return render_template(
        "eval.html",
        n_questions=len(questions),
        runs=list(reversed(runs)),      # newest first in history table
        selected=selected,
        sweep_json=json.dumps(sweep_data),
        rank_dist_json=json.dumps(rank_dist),
        default_max_k=DEFAULT_MAX_K,
    )


@app.route("/eval/run", methods=["POST"])
def eval_run():
    max_k = int(request.form.get("max_k", DEFAULT_MAX_K))
    max_k = max(1, min(max_k, 50))           # clamp to sane range

    questions   = load_questions(EVAL_PATH)
    k_values    = list(range(1, max_k + 1))
    sweep        = run_eval_sweep(questions, k_values)
    per_question = run_eval(questions, top_k=max_k, generate_answers=True)
    metrics     = aggregate_metrics(per_question)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = {
        "id":           run_id,
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "max_k":        max_k,
        "n_questions":  len(questions),
        "metrics":      metrics,           # headline numbers at max_k
        "sweep":        {str(k): v for k, v in sweep.items()},
        "per_question": per_question,
    }
    _save_run(run)

    return redirect(url_for("eval_dashboard", run=run_id))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
