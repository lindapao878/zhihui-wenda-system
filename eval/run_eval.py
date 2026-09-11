"""Run the golden-set evaluation against the local query graph.

Usage:
    python eval/run_eval.py --output eval/reports/eval_report.json
    python eval/run_eval.py --repeat 2 --limit 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import aggregate_results, load_cases, score_case  # noqa: E402
from knowledge.processor.query_process.main_graph import query_app  # noqa: E402
from knowledge.utils.query_cache import query_cache  # noqa: E402


def _invoke_case(case: Dict[str, Any]) -> Dict[str, Any]:
    cache_hit = query_cache.get(case["query"]) is not None
    state = {
        "original_query": case["query"],
        "session_id": f"eval-{case['id']}",
        "task_id": str(uuid.uuid4()),
        "is_stream": False,
        "history": case.get("history", []),
    }
    start_time = time.perf_counter()
    try:
        final_state = query_app.invoke(state)
        error = None
    except Exception as exc:  # Keep evaluation running after one bad case.
        final_state = {}
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - start_time) * 1000
    return score_case(case, final_state, latency_ms, cache_hit=cache_hit, error=error)


def run(args: argparse.Namespace) -> Path:
    cases = load_cases(args.input)
    if args.limit:
        cases = cases[: args.limit]

    quality_results: List[Dict[str, Any]] = []
    all_latencies: List[float] = []
    cache_hits = 0
    invocations = 0
    errors: List[Dict[str, str]] = []

    for case in cases:
        first_result = _invoke_case(case)
        quality_results.append(first_result)
        all_latencies.append(first_result["latency_ms"])
        invocations += 1
        cache_hits += 1 if first_result["cache_hit"] else 0
        if first_result["error"]:
            errors.append({"id": case["id"], "error": first_result["error"]})

        for _ in range(1, args.repeat):
            repeat_result = _invoke_case(case)
            all_latencies.append(repeat_result["latency_ms"])
            invocations += 1
            cache_hits += 1 if repeat_result["cache_hit"] else 0
            if repeat_result["error"]:
                errors.append({"id": case["id"], "error": repeat_result["error"], "repeat": "true"})

    from eval import percentile

    metrics = aggregate_results(quality_results)
    metrics["p95_latency_ms"] = percentile(all_latencies, 95)
    metrics["cache_hit_rate"] = round(cache_hits / invocations, 4) if invocations else None
    metrics["invocations"] = invocations
    metrics["repeat"] = args.repeat

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(args.input).resolve()),
        "metrics": metrics,
        "results": quality_results,
        "errors": errors,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the golden-set evaluation")
    parser.add_argument("--input", default=str(ROOT / "eval" / "golden.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "eval" / "reports" / "eval_report.json"))
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N cases")
    parser.add_argument("--repeat", type=int, default=1, help="Invoke each case N times for cache metrics")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    report_path = run(arguments)
    print(f"EVAL_REPORT={report_path}")
