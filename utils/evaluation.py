"""
Evaluation metrics for RAG systems.
Provides metrics for retrieval quality, answer relevance, and citation accuracy.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class RetrievalMetrics:
    """Metrics for document retrieval evaluation."""
    precision_at_k: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    hit_rate_at_k: float


@dataclass
class GenerationMetrics:
    """Metrics for answer generation evaluation."""
    answer_relevance_score: float
    citation_precision: float
    citation_recall: float
    source_groundedness: float
    completeness_score: float


@dataclass
class RAGEvaluationResult:
    """Complete RAG evaluation result."""
    query: str
    retrieved_documents: List[Dict]
    answer: str
    ground_truth: Optional[str] = None
    retrieval_metrics: Optional[RetrievalMetrics] = None
    generation_metrics: Optional[GenerationMetrics] = None
    latency_ms: float = 0.0


class RetrievalEvaluator:
    """Evaluator for retrieval quality."""

    def __init__(self, k_values: List[int] = None):
        """
        Initialize the retrieval evaluator.

        Args:
            k_values: List of k values for evaluation (e.g., [1, 5, 10])
        """
        self.k_values = k_values or [1, 3, 5, 10]

    def compute_precision_at_k(
        self,
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int
    ) -> float:
        """
        Compute precision at k.

        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            k: Number of top results to consider

        Returns:
            Precision at k
        """
        if k <= 0 or len(retrieved_docs) == 0:
            return 0.0

        top_k_retrieved = retrieved_docs[:k]
        relevant_set = set(relevant_docs)

        if len(top_k_retrieved) == 0:
            return 0.0

        hits = sum(1 for doc in top_k_retrieved if doc in relevant_set)
        return hits / k

    def compute_recall_at_k(
        self,
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int
    ) -> float:
        """
        Compute recall at k.

        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            k: Number of top results to consider

        Returns:
            Recall at k
        """
        if len(relevant_docs) == 0:
            return 1.0

        top_k_retrieved = retrieved_docs[:k]
        relevant_set = set(relevant_docs)

        hits = sum(1 for doc in top_k_retrieved if doc in relevant_set)
        return hits / len(relevant_set)

    def compute_mrr_at_k(
        self,
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int
    ) -> float:
        """
        Compute Mean Reciprocal Rank at k.

        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            k: Number of top results to consider

        Returns:
            MRR at k
        """
        relevant_set = set(relevant_docs)

        for i, doc in enumerate(retrieved_docs[:k], 1):
            if doc in relevant_set:
                return 1.0 / i

        return 0.0

    def compute_ndcg_at_k(
        self,
        relevant_docs: List[str],
        retrieved_docs: List[str],
        relevance_scores: Dict[str, float],
        k: int
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at k.

        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            relevance_scores: Dictionary mapping doc IDs to relevance scores
            k: Number of top results to consider

        Returns:
            NDCG at k
        """
        top_k_retrieved = retrieved_docs[:k]

        def dcg_at_k(retrieved):
            dcg = 0.0
            for i, doc in enumerate(retrieved, 1):
                relevance = relevance_scores.get(doc, 0)
                dcg += relevance / math.log2(i + 1)
            return dcg

        dcg = dcg_at_k(top_k_retrieved)

        ideal_order = sorted(relevant_docs, key=lambda x: relevance_scores.get(x, 0), reverse=True)
        ideal_dcg = dcg_at_k(ideal_order[:k])

        if ideal_dcg == 0:
            return 0.0

        return dcg / ideal_dcg

    def compute_hit_rate_at_k(
        self,
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int
    ) -> float:
        """
        Compute hit rate at k (whether any relevant doc is in top k).

        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            k: Number of top results to consider

        Returns:
            Hit rate at k
        """
        top_k_retrieved = retrieved_docs[:k]
        relevant_set = set(relevant_docs)

        hits = sum(1 for doc in top_k_retrieved if doc in relevant_set)
        return 1.0 if hits > 0 else 0.0

    def evaluate_retrieval(
        self,
        relevant_docs: List[str],
        retrieved_docs: List[str],
        relevance_scores: Dict[str, float] = None
    ) -> RetrievalMetrics:
        """
        Compute all retrieval metrics.

        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            relevance_scores: Optional relevance scores for NDCG

        Returns:
            RetrievalMetrics object
        """
        k = min(self.k_values)
        precision_k = self.compute_precision_at_k(relevant_docs, retrieved_docs, k)
        recall_k = self.compute_recall_at_k(relevant_docs, retrieved_docs, k)
        mrr_k = self.compute_mrr_at_k(relevant_docs, retrieved_docs, k)
        ndcg_k = self.compute_ndcg_at_k(
            relevant_docs, retrieved_docs,
            relevance_scores or {}, k
        )
        hit_rate_k = self.compute_hit_rate_at_k(relevant_docs, retrieved_docs, k)

        return RetrievalMetrics(
            precision_at_k=precision_k,
            recall_at_k=recall_k,
            mrr_at_k=mrr_k,
            ndcg_at_k=ndcg_k,
            hit_rate_at_k=hit_rate_k
        )


class GenerationEvaluator:
    """Evaluator for answer generation quality."""

    def __init__(self):
        """Initialize the generation evaluator."""
        pass

    def compute_answer_relevance(
        self,
        question: str,
        answer: str,
        retrieved_docs: List[str]
    ) -> float:
        """
        Compute answer relevance score.

        Args:
            question: Original question
            answer: Generated answer
            retrieved_docs: Retrieved documents used as context

        Returns:
            Relevance score between 0 and 1
        """
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        if len(answer_words) == 0:
            return 0.0

        overlap = len(question_words & answer_words)
        base_relevance = overlap / len(question_words)

        answer_length = len(answer.split())
        expected_length = max(10, len(question.split()) * 2)

        if answer_length < expected_length * 0.5:
            length_penalty = 0.5
        elif answer_length < expected_length:
            length_penalty = 0.7
        else:
            length_penalty = 1.0

        context_mentions = 0
        for doc in retrieved_docs:
            if any(word in answer.lower() for word in doc[:200].lower().split()):
                context_mentions += 1

        context_score = context_mentions / len(retrieved_docs) if retrieved_docs else 0

        return (base_relevance * 0.3 + context_score * 0.5 + length_penalty * 0.2)

    def compute_citation_precision(
        self,
        answer: str,
        retrieved_docs: List[Dict]
    ) -> float:
        """
        Compute citation precision (how many cited sources are relevant).

        Args:
            answer: Generated answer
            retrieved_docs: Retrieved documents with metadata

        Returns:
            Citation precision score
        """
        import re

        cited_codes = set()
        cited_articles = set()

        for doc in retrieved_docs:
            code = doc.get("metadata", {}).get("code", "")
            article = doc.get("metadata", {}).get("article", "")
            if code:
                cited_codes.add(code.lower())
            if article:
                cited_articles.add(article.lower())

        if not cited_codes and not cited_articles:
            return 0.5

        citation_patterns = [
            r"art\.?\s*([a-z0-9\-]+)",
            r"article\s+([a-z0-9\-]+)",
            r"code\s+([a-z]+)",
        ]

        found_codes = set()
        found_articles = set()

        for pattern in citation_patterns:
            matches = re.findall(pattern, answer.lower())
            for match in matches:
                if match in cited_codes:
                    found_codes.add(match)
                if match in cited_articles:
                    found_articles.add(match)

        total_citations = len(found_codes) + len(found_articles)
        total_possible = len(cited_codes) + len(cited_articles)

        if total_possible == 0:
            return 0.0

        return min(1.0, total_citations / max(1, total_possible * 0.5))

    def compute_citation_recall(
        self,
        answer: str,
        retrieved_docs: List[Dict],
        ground_truth_citations: List[str] = None
    ) -> float:
        """
        Compute citation recall.

        Args:
            answer: Generated answer
            retrieved_docs: Retrieved documents
            ground_truth_citations: Optional list of ground truth citations

        Returns:
            Citation recall score
        """
        import re

        answer_citations = set()
        citation_patterns = [
            r"art\.?\s*([a-z0-9\-]+)",
            r"article\s+([a-z0-9\-]+)",
        ]

        for pattern in citation_patterns:
            matches = re.findall(pattern, answer.lower())
            answer_citations.update(matches)

        if ground_truth_citations:
            ground_set = set(gt.lower() for gt in ground_truth_citations)
            if len(ground_set) == 0:
                return 1.0
            found = len(answer_citations & ground_set)
            return found / len(ground_set)

        return 1.0 if len(answer_citations) > 0 else 0.5

    def compute_source_groundedness(
        self,
        answer: str,
        retrieved_docs: List[Dict]
    ) -> float:
        """
        Compute how well the answer is grounded in the sources.

        Args:
            answer: Generated answer
            retrieved_docs: Retrieved documents

        Returns:
            Groundedness score
        """
        import re

        answer_lower = answer.lower()

        source_evidence = []
        for doc in retrieved_docs:
            doc_text = doc.get("text", doc.get("document", ""))
            doc_lower = doc_text.lower()

            doc_evidence = []
            for sentence in re.split(r'[.!?]', answer_lower):
                sentence = sentence.strip()
                if len(sentence) > 10:
                    overlap = len(set(sentence.split()) & set(doc_lower.split()))
                    doc_evidence.append(overlap / len(sentence.split()))

            if doc_evidence:
                source_evidence.append(max(doc_evidence))

        if not source_evidence:
            return 0.0

        return min(1.0, sum(source_evidence) / len(source_evidence))

    def compute_completeness(
        self,
        question: str,
        answer: str,
        retrieved_docs: List[Dict]
    ) -> float:
        """
        Compute answer completeness.

        Args:
            question: Original question
            answer: Generated answer
            retrieved_docs: Retrieved documents

        Returns:
            Completeness score
        """
        question_lower = question.lower()

        aspects = []
        if any(word in question_lower for word in ['comment', 'quoi', 'quel']):
            aspects.append('procedure')
        if any(word in question_lower for word in ['condition', 'exiger', 'peut']):
            aspects.append('conditions')
        if any(word in question_lower for word in ['droit', 'obligat', 'devoir']):
            aspects.append('rights')
        if any(word in question_lower for word in ['quand', 'date', 'délai']):
            aspects.append('timing')
        if any(word in question_lower for word in ['pourquoi', 'raison']):
            aspects.append('reason')

        if not aspects:
            return 0.7

        aspects_covered = 0
        answer_lower = answer.lower()

        if 'procedure' in aspects:
            if any(word in answer_lower for word in ['doit', 'faut', 'il faut', 'il est']):
                aspects_covered += 1
        if 'conditions' in aspects:
            if any(word in answer_lower for word in ['si', 'lorsque', 'quand', 'condition']):
                aspects_covered += 1
        if 'rights' in aspects:
            if any(word in answer_lower for word in ['droit', 'obligat', 'peut']):
                aspects_covered += 1
        if 'timing' in aspects:
            if any(word in answer_lower for word in ['jour', 'mois', 'an', 'délai', 'date']):
                aspects_covered += 1
        if 'reason' in aspects:
            if any(word in answer_lower for word in ['parce que', 'afin', 'cause', 'raison']):
                aspects_covered += 1

        return aspects_covered / len(aspects)

    def evaluate_generation(
        self,
        question: str,
        answer: str,
        retrieved_docs: List[Dict],
        ground_truth_citations: List[str] = None
    ) -> GenerationMetrics:
        """
        Compute all generation metrics.

        Args:
            question: Original question
            answer: Generated answer
            retrieved_docs: Retrieved documents
            ground_truth_citations: Optional ground truth citations

        Returns:
            GenerationMetrics object
        """
        doc_ids = [
            doc.get("metadata", {}).get("id", str(i))
            for i, doc in enumerate(retrieved_docs)
        ]

        return GenerationMetrics(
            answer_relevance_score=self.compute_answer_relevance(
                question, answer, doc_ids
            ),
            citation_precision=self.compute_citation_precision(answer, retrieved_docs),
            citation_recall=self.compute_citation_recall(
                answer, retrieved_docs, ground_truth_citations
            ),
            source_groundedness=self.compute_source_groundedness(answer, retrieved_docs),
            completeness_score=self.compute_completeness(
                question, answer, retrieved_docs
            )
        )


class RAGEvaluator:
    """Complete RAG system evaluator."""

    def __init__(self):
        """Initialize the RAG evaluator."""
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()

    def evaluate(
        self,
        query: str,
        retrieved_docs: List[Dict],
        answer: str,
        relevant_docs: List[str] = None,
        relevance_scores: Dict[str, float] = None,
        ground_truth: str = None,
        ground_truth_citations: List[str] = None,
        latency_ms: float = 0.0
    ) -> RAGEvaluationResult:
        """
        Complete RAG evaluation.

        Args:
            query: User query
            retrieved_docs: Retrieved documents
            answer: Generated answer
            relevant_docs: Relevant document IDs for retrieval eval
            relevance_scores: Relevance scores for NDCG
            ground_truth: Ground truth answer
            ground_truth_citations: Ground truth citations
            latency_ms: Response latency in milliseconds

        Returns:
            RAGEvaluationResult object
        """
        retrieval_metrics = None
        if relevant_docs is not None:
            doc_ids = [
                doc.get("metadata", {}).get("id", str(i))
                for i, doc in enumerate(retrieved_docs)
            ]
            retrieval_metrics = self.retrieval_evaluator.evaluate_retrieval(
                relevant_docs, doc_ids, relevance_scores
            )

        generation_metrics = self.generation_evaluator.evaluate_generation(
            query, answer, retrieved_docs, ground_truth_citations
        )

        return RAGEvaluationResult(
            query=query,
            retrieved_documents=retrieved_docs,
            answer=answer,
            ground_truth=ground_truth,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            latency_ms=latency_ms
        )

    def aggregate_results(self, results: List[RAGEvaluationResult]) -> Dict:
        """
        Aggregate evaluation results across multiple queries.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary with aggregated metrics
        """
        if not results:
            return {}

        agg = {
            "total_queries": len(results),
            "avg_latency_ms": sum(r.latency_ms for r in results) / len(results),
        }

        if results[0].retrieval_metrics:
            for metric in ["precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k", "hit_rate_at_k"]:
                values = [getattr(r.retrieval_metrics, metric) for r in results if r.retrieval_metrics]
                if values:
                    agg[f"avg_{metric}"] = sum(values) / len(values)

        if results[0].generation_metrics:
            for metric in [
                "answer_relevance_score", "citation_precision", "citation_recall",
                "source_groundedness", "completeness_score"
            ]:
                values = [getattr(r.generation_metrics, metric) for r in results if r.generation_metrics]
                if values:
                    agg[f"avg_{metric}"] = sum(values) / len(values)

        return agg


if __name__ == "__main__":
    evaluator = RAGEvaluator()

    retrieved_docs = [
        {
            "text": "Article L1234-1 du Code du travail: La période d'essai permet aux parties d'évaluer les conditions de travail.",
            "metadata": {"id": "doc1", "code": "Code du travail", "article": "L1234-1"}
        },
        {
            "text": "Article L1234-2 du Code du travail: La durée de la période d'essai est déterminée par la loi.",
            "metadata": {"id": "doc2", "code": "Code du travail", "article": "L1234-2"}
        },
    ]

    result = evaluator.evaluate(
        query="Quelles sont les conditions de la période d'essai?",
        retrieved_docs=retrieved_docs,
        answer="Selon l'Article L1234-1 du Code du travail, la période d'essai permet aux parties d'évaluer les conditions de travail. L'Article L1234-2 précise que sa durée est déterminée par la loi.",
        relevant_docs=["doc1", "doc2"],
        latency_ms=150.0
    )

    print("Evaluation Result:")
    print(f"Query: {result.query}")
    print(f"Latency: {result.latency_ms}ms")
    print(f"Answer Relevance: {result.generation_metrics.answer_relevance_score:.3f}")
    print(f"Citation Precision: {result.generation_metrics.citation_precision:.3f}")
    print(f"Source Groundedness: {result.generation_metrics.source_groundedness:.3f}")
    print(f"Completeness: {result.generation_metrics.completeness_score:.3f}")
