"""Evaluation package for the query pipeline."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def load_cases(path: str) -> List[Dict[str, Any]]:
    """Load and minimally validate JSONL cases."""
    import json
    from pathlib import Path

    cases: List[Dict[str, Any]] = []
    seen_ids = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            required = {"id", "scenario", "query"}
            missing = required - set(case)
            if missing:
                raise ValueError(f"missing {sorted(missing)} at {path}:{line_number}")
            if case["id"] in seen_ids:
                raise ValueError(f"duplicate id {case['id']} at {path}:{line_number}")
            seen_ids.add(case["id"])
            cases.append(case)
    return cases


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def keyword_coverage(text: str, keywords: Optional[List[str]]) -> Optional[float]:
    """Return the fraction of expected keywords present in text."""
    if not keywords:
        return None
    normalized_text = _normalize(text)
    matched = sum(1 for keyword in keywords if _normalize(keyword) in normalized_text)
    return matched / len(keywords)


def _doc_reference(document: Dict[str, Any]) -> str:
    entity = document.get("entity") if isinstance(document.get("entity"), dict) else {}
    if document.get("source"):
        return str(document["source"])
    if document.get("file_title"):
        return str(document["file_title"])
    if document.get("parent_title"):
        return str(document["parent_title"])
    if document.get("title"):
        return str(document["title"])
    if entity.get("file_title"):
        return str(entity["file_title"])
    if entity.get("title"):
        return str(entity["title"])
    return ""


def source_references(documents: List[Dict[str, Any]]) -> List[str]:
    return [_normalize(_doc_reference(document)) for document in documents or []]


def _reference_hit_ratio(expected: List[str], references: List[str]) -> Optional[float]:
    if not expected:
        return None
    reference_set = set(references)
    matched = sum(1 for item in expected if _normalize(item) in reference_set)
    return matched / len(expected)


def _refusal_probability(answer: str) -> bool:
    normalized = _normalize(answer)
    markers = ("抱歉", "无法", "不能", "不确定", "没有找到", "未能找到", "无法回答", "知识库中")
    return any(marker in normalized for marker in markers)


def score_case(
    case: Dict[str, Any],
    state: Dict[str, Any],
    latency_ms: float,
    cache_hit: bool = False,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Score one query result without calling another model."""
    answer = str(state.get("answer", ""))
    documents = [doc for doc in state.get("reranked_docs", []) if isinstance(doc, dict)]
    top1_docs = documents[:1]
    top3_docs = documents[:3]
    expected_chunks = [str(item) for item in case.get("expected_chunk_ids", [])]
    expected_titles = [str(item) for item in case.get("expected_file_titles", [])]

    if expected_chunks:
        chunk_ids = [str(doc.get("chunk_id", "")) for doc in documents]
        top1_recall = _reference_hit_ratio(expected_chunks, chunk_ids[:1])
        top3_recall = _reference_hit_ratio(expected_chunks, chunk_ids[:3])
        citation_ratio = _reference_hit_ratio(expected_chunks, chunk_ids[: max(1, len(documents))])
    elif expected_titles:
        top1_refs = source_references(top1_docs)
        top3_refs = source_references(top3_docs)
        all_refs = source_references(documents)
        top1_recall = _reference_hit_ratio(expected_titles, top1_refs)
        top3_recall = _reference_hit_ratio(expected_titles, top3_refs)
        explicit_refs = state.get("source_refs") or state.get("item_names") or []
        citation_ratio = _reference_hit_ratio(expected_titles, [_normalize(ref) for ref in explicit_refs] or all_refs)
    else:
        top1_context = " ".join(str(doc.get("content", "")) for doc in top1_docs)
        top3_context = " ".join(str(doc.get("content", "")) for doc in top3_docs)
        top1_recall = keyword_coverage(top1_context, case.get("expected_answer_keywords"))
        top3_recall = keyword_coverage(top3_context, case.get("expected_answer_keywords"))
        citation_ratio = top3_recall

    faithfulness = keyword_coverage(answer, case.get("expected_answer_keywords"))
    if case.get("expect_refusal"):
        faithfulness = 1.0 if _refusal_probability(answer) else 0.0

    expected_images = case.get("expected_image_count")
    image_count = len(state.get("image_urls", []) or [])
    if expected_images is None:
        image_correct = None
    else:
        image_correct = image_count == int(expected_images)

    return {
        "id": case["id"],
        "scenario": case["scenario"],
        "query": case["query"],
        "answer": answer,
        "top1_recall": top1_recall,
        "top3_recall": top3_recall,
        "faithfulness": faithfulness,
        "citation_completeness": citation_ratio,
        "latency_ms": float(latency_ms),
        "cache_hit": bool(cache_hit),
        "image_count": image_count,
        "image_correct": image_correct,
        "error": error,
    }


def _mean(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def percentile(values: List[float], percentile_value: float) -> Optional[float]:
    """Nearest-rank percentile, stable for small evaluation sets."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, round(percentile_value / 100 * len(ordered)))
    return ordered[rank - 1]


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-case scores into the report-level metrics."""
    latencies = [result["latency_ms"] for result in results if result["latency_ms"] >= 0]

    def values(field: str) -> List[float]:
        return [result[field] for result in results if result.get(field) is not None]

    scenarios: Dict[str, Dict[str, Any]] = {}
    for scenario in sorted({result["scenario"] for result in results}):
        scenario_results = [result for result in results if result["scenario"] == scenario]
        scenarios[scenario] = {
            "case_count": len(scenario_results),
            "top1_recall": _mean(values_for(scenario_results, "top1_recall")),
            "top3_recall": _mean(values_for(scenario_results, "top3_recall")),
            "faithfulness": _mean(values_for(scenario_results, "faithfulness")),
            "citation_completeness": _mean(values_for(scenario_results, "citation_completeness")),
            "p95_latency_ms": percentile(
                [result["latency_ms"] for result in scenario_results],
                95,
            ),
        }

    return {
        "case_count": len(results),
        "top1_recall": _mean(values("top1_recall")),
        "top3_recall": _mean(values("top3_recall")),
        "faithfulness": _mean(values("faithfulness")),
        "citation_completeness": _mean(values("citation_completeness")),
        "p95_latency_ms": percentile(latencies, 95),
        "cache_hit_rate": _mean([1.0 if result.get("cache_hit") else 0.0 for result in results]),
        "scenario_metrics": scenarios,
    }


def values_for(results: List[Dict[str, Any]], field: str) -> List[float]:
    return [result[field] for result in results if result.get(field) is not None]
