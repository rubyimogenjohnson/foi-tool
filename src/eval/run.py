"""
eval/run.py — retrieval evaluation over ground-truth question–answer pairs.

Metrics computed per question, then averaged:
  hit@k        — 1 if any relevant identifier appears in the top-k results
  precision@k  — (# relevant in top-k) / k
  recall@k     — (# relevant in top-k) / (# relevant total for the question)
  reciprocal_rank — 1 / rank of the first relevant result (0 if not found)
  rank         — rank of the first relevant result (None if not found)

Run from the project root:
    python -m src.eval.run
    python -m src.eval.run data/eval_questions.json --top-k 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import anthropic

from src.retrieval.format import format_context
from src.retrieval.search import search

_EVAL_SYSTEM = """\
You are evaluating a Camden Council FOI search tool.
Answer the question using ONLY the provided FOI excerpts.
For every claim cite the FOI reference number and date, e.g. "According to CAM10600 (12 March 2024), …".
If nothing in the excerpts is relevant say so clearly. Be concise (3–5 sentences).
"""


def _llm_answer(question: str, hits: list[dict]) -> str:
    if not hits:
        return "(no results returned — cannot generate answer)"
    try:
        context = format_context(hits)
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=_EVAL_SYSTEM,
            messages=[{"role": "user", "content": f"Question: {question}\n\nRelevant FOI excerpts:\n\n{context}"}],
        )
        return msg.content[0].text
    except Exception as exc:
        return f"(LLM error: {exc})"

DEFAULT_EVAL_PATH = Path(__file__).parents[2] / "data" / "eval_questions.json"
DEFAULT_TOP_K = 10


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    for q in questions:
        if "relevant_identifiers" not in q or not q["relevant_identifiers"]:
            raise ValueError(f"Question {q.get('id', '?')} has no relevant_identifiers")
    return questions


def _first_relevant_rank(result_ids: list[str], relevant: set[str]) -> int | None:
    """Return the 1-based rank of the first relevant result, or None."""
    for rank, rid in enumerate(result_ids, start=1):
        if rid in relevant:
            return rank
    return None


def evaluate_question(question: dict, top_k: int, generate_answer: bool = False) -> dict:
    """Run search for one question and return per-question metrics."""
    q_text = question["question"]
    relevant = set(question["relevant_identifiers"])

    hits = search(q_text, top_k=top_k, recency_boost=0.0)
    result_ids = [h["identifier"] for h in hits]

    found_relevant = relevant & set(result_ids)
    first_rank = _first_relevant_rank(result_ids, relevant)

    hit = int(bool(found_relevant))
    precision = len(found_relevant) / top_k if top_k > 0 else 0.0
    recall = len(found_relevant) / len(relevant) if relevant else 0.0
    rr = (1.0 / first_rank) if first_rank is not None else 0.0

    result = {
        "id": question.get("id", ""),
        "question": q_text,
        "relevant_identifiers": list(relevant),
        "result_identifiers": result_ids,
        "hit": hit,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "reciprocal_rank": round(rr, 4),
        "rank": first_rank,  # None means not found within top_k
        "llm_answer": None,
    }

    if generate_answer:
        result["llm_answer"] = _llm_answer(q_text, hits)

    return result


def run_eval(questions: list[dict], top_k: int = DEFAULT_TOP_K, generate_answers: bool = False) -> list[dict]:
    """Evaluate all questions and return per-question result dicts."""
    results = []
    for i, q in enumerate(questions, start=1):
        label = q['question']
        if len(label) > 70:
            label = label[:67] + "…"
        print(f"  [{i}/{len(questions)}] {label}")
        results.append(evaluate_question(q, top_k, generate_answer=generate_answers))
    return results


def aggregate_metrics(results: list[dict]) -> dict:
    """Compute mean metrics across all questions."""
    n = len(results)
    if n == 0:
        return {}

    ranks_found = [r["rank"] for r in results if r["rank"] is not None]

    return {
        "n_questions": n,
        "hit_rate": round(sum(r["hit"] for r in results) / n, 4),
        "mean_precision": round(sum(r["precision"] for r in results) / n, 4),
        "mean_recall": round(sum(r["recall"] for r in results) / n, 4),
        "mrr": round(sum(r["reciprocal_rank"] for r in results) / n, 4),
        "mean_rank": round(sum(ranks_found) / len(ranks_found), 2) if ranks_found else None,
        "found_rate": round(len(ranks_found) / n, 4),
    }


def _print_summary(metrics: dict) -> None:
    print("\n=== Evaluation Summary ===")
    print(f"  Questions evaluated : {metrics['n_questions']}")
    print(f"  Hit rate            : {metrics['hit_rate']:.1%}")
    print(f"  Mean Precision@K    : {metrics['mean_precision']:.4f}")
    print(f"  Mean Recall@K       : {metrics['mean_recall']:.4f}")
    print(f"  MRR                 : {metrics['mrr']:.4f}")
    if metrics["mean_rank"] is not None:
        print(f"  Mean Rank           : {metrics['mean_rank']:.2f}")
    else:
        print(f"  Mean Rank           : N/A (no results found)")
    print(f"  Found rate          : {metrics['found_rate']:.1%}")


def run_eval_sweep(questions: list[dict], k_values: list[int]) -> dict[int, dict]:
    """
    Run evaluation at multiple K values.
    Returns {k: aggregate_metrics} for plotting precision@k / recall@k curves.
    """
    sweep: dict[int, dict] = {}
    max_k = max(k_values)
    print(f"Running sweep (max K={max_k})…")

    # Fetch at max K once per question, then slice for smaller K
    raw_results: list[dict] = []
    for i, q in enumerate(questions, start=1):
        label = q['question']
        if len(label) > 70:
            label = label[:67] + "…"
        print(f"  [{i}/{len(questions)}] {label}")
        raw_results.append(evaluate_question(q, max_k))

    for k in k_values:
        sliced = []
        for r in raw_results:
            result_ids_at_k = r["result_identifiers"][:k]
            relevant = set(r["relevant_identifiers"])
            found = relevant & set(result_ids_at_k)
            first_rank = _first_relevant_rank(result_ids_at_k, relevant)
            hit = int(bool(found))
            sliced.append({
                "id": r["id"],
                "question": r["question"],
                "relevant_identifiers": r["relevant_identifiers"],
                "result_identifiers": result_ids_at_k,
                "hit": hit,
                "precision": round(len(found) / k, 4),
                "recall": round(len(found) / len(relevant) if relevant else 0.0, 4),
                "reciprocal_rank": round(1.0 / first_rank if first_rank else 0.0, 4),
                "rank": first_rank,
            })
        sweep[k] = aggregate_metrics(sliced)

    return sweep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument("eval_path", nargs="?", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--out", type=Path, help="Write JSON results to this path")
    args = parser.parse_args()

    questions = load_questions(args.eval_path)
    print(f"Loaded {len(questions)} questions from {args.eval_path}")

    results = run_eval(questions, top_k=args.top_k)
    metrics = aggregate_metrics(results)
    _print_summary(metrics)

    if args.out:
        payload = {"metrics": metrics, "results": results}
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nResults written to {args.out}")
