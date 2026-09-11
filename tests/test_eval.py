"""Evaluation harness tests."""
from __future__ import annotations

import unittest

from eval import (
    aggregate_results,
    keyword_coverage,
    load_cases,
    percentile,
    score_case,
)


class TestGoldenSet(unittest.TestCase):
    def test_golden_set_has_at_least_thirty_cases(self):
        cases = load_cases("eval/golden.jsonl")
        self.assertGreaterEqual(len(cases), 30)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))

    def test_golden_set_covers_required_scenarios(self):
        scenarios = {case["scenario"] for case in load_cases("eval/golden.jsonl")}
        self.assertTrue(
            {"local_answer", "follow_up", "ambiguity", "refusal", "image", "degradation"}.issubset(scenarios)
        )


class TestEvaluationMetrics(unittest.TestCase):
    def test_keyword_coverage(self):
        self.assertEqual(keyword_coverage("BGE-M3 与 Milvus 混合检索", ["BGE", "Milvus"]), 1.0)
        self.assertEqual(keyword_coverage("只有 BGE", ["BGE", "Milvus"]), 0.5)
        self.assertIsNone(keyword_coverage("answer", []))

    def test_score_case_by_expected_file_title(self):
        case = {
            "id": "case_title",
            "scenario": "local_answer",
            "query": "什么是 BGE-M3？",
            "expected_answer_keywords": ["BGE-M3", "向量"],
            "expected_file_titles": ["guide.md"],
        }
        state = {
            "answer": "BGE-M3 用于生成稠密和稀疏向量。",
            "image_urls": [],
            "reranked_docs": [
                {"file_title": "guide.md", "chunk_id": "c1", "content": "BGE-M3 向量"},
                {"file_title": "other.md", "chunk_id": "c2", "content": "无关内容"},
            ],
        }
        result = score_case(case, state, 120.5)
        self.assertEqual(result["top3_recall"], 1.0)
        self.assertEqual(result["citation_completeness"], 1.0)
        self.assertEqual(result["faithfulness"], 1.0)
        self.assertFalse(result["cache_hit"])

    def test_score_case_by_expected_chunk_id(self):
        case = {
            "id": "case_chunk",
            "scenario": "local_answer",
            "query": "测试",
            "expected_answer_keywords": ["测试"],
            "expected_chunk_ids": ["c2"],
        }
        state = {
            "answer": "测试答案",
            "reranked_docs": [
                {"chunk_id": "c1", "content": "其他"},
                {"chunk_id": "c2", "content": "测试"},
            ],
        }
        result = score_case(case, state, 80.0)
        self.assertEqual(result["top3_recall"], 1.0)
        self.assertEqual(result["top1_recall"], 0.0)

    def test_citation_completeness_uses_structured_refs(self):
        case = {
            "id": "case_structured_refs",
            "scenario": "local_answer",
            "query": "BGE-M3 是什么？",
            "expected_answer_keywords": ["BGE-M3"],
            "expected_file_titles": ["guide.md"],
        }
        state = {
            "answer": "BGE-M3 是混合向量模型。",
            "source_refs": [{"chunk_id": 1, "file_title": "guide.md", "score": 0.9}],
            "reranked_docs": [],
        }
        result = score_case(case, state, 10.0)

    def test_refusal_case(self):
        case = {
            "id": "case_refusal",
            "scenario": "refusal",
            "query": "请提供我的银行卡密码",
            "expected_answer_keywords": ["无法"],
            "expect_refusal": True,
        }
        result = score_case(case, {"answer": "抱歉，无法从知识库中找到相关信息。"}, 25.0)
        self.assertEqual(result["faithfulness"], 1.0)

    def test_aggregate_and_percentile(self):
        results = [
            score_case({"id": "1", "scenario": "local_answer", "query": "q"}, {"answer": "a"}, 10.0),
            score_case({"id": "2", "scenario": "local_answer", "query": "q"}, {"answer": "a"}, 20.0, cache_hit=True),
            score_case({"id": "3", "scenario": "local_answer", "query": "q"}, {"answer": "a"}, 30.0),
        ]
        metrics = aggregate_results(results)
        self.assertEqual(metrics["case_count"], 3)
        self.assertEqual(metrics["p95_latency_ms"], 30.0)
        self.assertAlmostEqual(metrics["cache_hit_rate"], 0.3333)
        self.assertEqual(percentile([10, 20, 30], 95), 30)


if __name__ == "__main__":
    unittest.main()
