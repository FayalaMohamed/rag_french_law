"""
Tests for evaluation metrics.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.evaluation import (
    RetrievalEvaluator,
    GenerationEvaluator,
    RAGEvaluator,
    RetrievalMetrics,
    GenerationMetrics
)


class TestRetrievalEvaluator(unittest.TestCase):
    """Test cases for RetrievalEvaluator."""

    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = RetrievalEvaluator(k_values=[1, 3, 5])

    def test_precision_at_k(self):
        """Test precision at k calculation."""
        relevant_docs = ["doc1", "doc2", "doc3"]
        retrieved_docs = ["doc1", "doc4", "doc5"]
        precision = self.evaluator.compute_precision_at_k(relevant_docs, retrieved_docs, k=3)
        assert precision == 1/3

    def test_precision_at_k_zero_hits(self):
        """Test precision at k with no hits."""
        relevant_docs = ["doc1", "doc2"]
        retrieved_docs = ["doc3", "doc4", "doc5"]
        precision = self.evaluator.compute_precision_at_k(relevant_docs, retrieved_docs, k=3)
        assert precision == 0.0

    def test_recall_at_k(self):
        """Test recall at k calculation."""
        relevant_docs = ["doc1", "doc2", "doc3"]
        retrieved_docs = ["doc1", "doc4", "doc5"]
        recall = self.evaluator.compute_recall_at_k(relevant_docs, retrieved_docs, k=3)
        assert recall == 1/3

    def test_recall_at_k_all_found(self):
        """Test recall at k when all relevant docs are found."""
        relevant_docs = ["doc1", "doc2"]
        retrieved_docs = ["doc1", "doc2", "doc3"]
        recall = self.evaluator.compute_recall_at_k(relevant_docs, retrieved_docs, k=5)
        assert recall == 1.0

    def test_mrr_at_k(self):
        """Test MRR calculation."""
        relevant_docs = ["doc1", "doc2"]
        retrieved_docs = ["doc3", "doc1", "doc4"]
        mrr = self.evaluator.compute_mrr_at_k(relevant_docs, retrieved_docs, k=3)
        assert mrr == 1/2

    def test_mrr_at_k_no_hit(self):
        """Test MRR when no relevant doc is found."""
        relevant_docs = ["doc1"]
        retrieved_docs = ["doc2", "doc3", "doc4"]
        mrr = self.evaluator.compute_mrr_at_k(relevant_docs, retrieved_docs, k=3)
        assert mrr == 0.0

    def test_hit_rate_at_k(self):
        """Test hit rate calculation."""
        relevant_docs = ["doc1"]
        retrieved_docs = ["doc2", "doc3", "doc1"]
        hit_rate = self.evaluator.compute_hit_rate_at_k(relevant_docs, retrieved_docs, k=3)
        assert hit_rate == 1.0

    def test_hit_rate_at_k_miss(self):
        """Test hit rate when no hit."""
        relevant_docs = ["doc1"]
        retrieved_docs = ["doc2", "doc3", "doc4"]
        hit_rate = self.evaluator.compute_hit_rate_at_k(relevant_docs, retrieved_docs, k=3)
        assert hit_rate == 0.0

    def test_ndcg_at_k(self):
        """Test NDCG calculation."""
        relevant_docs = ["doc1", "doc2", "doc3"]
        retrieved_docs = ["doc3", "doc1", "doc2"]
        relevance_scores = {"doc1": 1.0, "doc2": 0.8, "doc3": 0.6}
        ndcg = self.evaluator.compute_ndcg_at_k(
            relevant_docs, retrieved_docs, relevance_scores, k=3
        )
        assert 0 <= ndcg <= 1.0

    def test_evaluate_retrieval(self):
        """Test complete retrieval evaluation."""
        relevant_docs = ["doc1", "doc2"]
        retrieved_docs = ["doc1", "doc3", "doc4"]
        metrics = self.evaluator.evaluate_retrieval(relevant_docs, retrieved_docs)
        assert isinstance(metrics, RetrievalMetrics)
        assert metrics.precision_at_k >= 0
        assert metrics.recall_at_k >= 0


class TestGenerationEvaluator(unittest.TestCase):
    """Test cases for GenerationEvaluator."""

    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = GenerationEvaluator()

    def test_compute_answer_relevance(self):
        """Test answer relevance calculation."""
        question = "What is the trial period?"
        answer = "The trial period is a test period."
        retrieved_docs = ["doc1", "doc2"]
        relevance = self.evaluator.compute_answer_relevance(
            question, answer, retrieved_docs
        )
        assert 0 <= relevance <= 1.0

    def test_compute_citation_precision(self):
        """Test citation precision calculation."""
        answer = "According to Art. L1234-1, the trial period is defined."
        retrieved_docs = [
            {
                "text": "Article L1234-1 du Code du travail...",
                "metadata": {"code": "travail", "article": "L1234-1"}
            }
        ]
        precision = self.evaluator.compute_citation_precision(answer, retrieved_docs)
        assert 0 <= precision <= 1.0

    def test_compute_citation_recall(self):
        """Test citation recall calculation."""
        answer = "According to Art. L1234-1, the trial period is defined."
        retrieved_docs = [
            {
                "text": "Article L1234-1 du Code du travail...",
                "metadata": {"code": "travail", "article": "L1234-1"}
            }
        ]
        recall = self.evaluator.compute_citation_recall(answer, retrieved_docs)
        assert 0 <= recall <= 1.0

    def test_compute_source_groundedness(self):
        """Test source groundedness calculation."""
        answer = "The trial period is defined in Article L1234-1."
        retrieved_docs = [
            {
                "text": "Article L1234-1 du Code du travail: The trial period is defined.",
                "metadata": {}
            }
        ]
        groundedness = self.evaluator.compute_source_groundedness(answer, retrieved_docs)
        assert 0 <= groundedness <= 1.0

    def test_compute_completeness(self):
        """Test completeness calculation."""
        question = "What are the conditions and procedures for termination?"
        answer = "The conditions for termination include valid reasons. The procedure requires notice."
        retrieved_docs = [
            {"text": "Conditions for termination...", "metadata": {}},
            {"text": "Procedure for termination...", "metadata": {}}
        ]
        completeness = self.evaluator.compute_completeness(
            question, answer, retrieved_docs
        )
        assert 0 <= completeness <= 1.0

    def test_evaluate_generation(self):
        """Test complete generation evaluation."""
        question = "What is the trial period?"
        answer = "The trial period is a test period."
        retrieved_docs = [
            {
                "text": "Article L1234-1 du Code du travail...",
                "metadata": {"id": "doc1"}
            }
        ]
        metrics = self.evaluator.evaluate_generation(
            question, answer, retrieved_docs
        )
        assert isinstance(metrics, GenerationMetrics)
        assert 0 <= metrics.answer_relevance_score <= 1.0
        assert 0 <= metrics.citation_precision <= 1.0


class TestRAGEvaluator(unittest.TestCase):
    """Test cases for RAGEvaluator."""

    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = RAGEvaluator()

    def test_evaluate_complete(self):
        """Test complete RAG evaluation."""
        query = "What are the conditions for trial period?"
        retrieved_docs = [
            {
                "text": "Article L1234-1: The trial period allows evaluation.",
                "metadata": {"id": "doc1"}
            }
        ]
        answer = "According to Article L1234-1, the trial period allows evaluation."

        result = self.evaluator.evaluate(
            query=query,
            retrieved_docs=retrieved_docs,
            answer=answer,
            relevant_docs=["doc1"],
            latency_ms=100.0
        )

        assert result.query == query
        assert result.answer == answer
        assert result.latency_ms == 100.0
        assert result.retrieval_metrics is not None
        assert result.generation_metrics is not None

    def test_aggregate_results(self):
        """Test aggregating evaluation results."""
        from utils.evaluation import RAGEvaluationResult, RetrievalMetrics

        results = [
            RAGEvaluationResult(
                query="Query 1",
                retrieved_documents=[],
                answer="Answer 1",
                retrieval_metrics=RetrievalMetrics(
                    precision_at_k=0.5, recall_at_k=0.6, mrr_at_k=0.7,
                    ndcg_at_k=0.8, hit_rate_at_k=0.9
                ),
                generation_metrics=GenerationMetrics(
                    answer_relevance_score=0.6, citation_precision=0.7,
                    citation_recall=0.8, source_groundedness=0.9,
                    completeness_score=0.5
                ),
                latency_ms=100.0
            ),
            RAGEvaluationResult(
                query="Query 2",
                retrieved_documents=[],
                answer="Answer 2",
                retrieval_metrics=RetrievalMetrics(
                    precision_at_k=0.7, recall_at_k=0.8, mrr_at_k=0.9,
                    ndcg_at_k=0.6, hit_rate_at_k=0.5
                ),
                generation_metrics=GenerationMetrics(
                    answer_relevance_score=0.8, citation_precision=0.9,
                    citation_recall=0.6, source_groundedness=0.7,
                    completeness_score=0.6
                ),
                latency_ms=150.0
            ),
        ]

        aggregated = self.evaluator.aggregate_results(results)

        assert aggregated["total_queries"] == 2
        assert aggregated["avg_latency_ms"] == 125.0
        assert "avg_precision_at_k" in aggregated
        assert "avg_answer_relevance_score" in aggregated
