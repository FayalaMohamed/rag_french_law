"""
Utility modules for French Legal RAG.
"""

from .text_processing import (
    LegalTextProcessor,
    chunk_documents,
    extract_citations,
    normalize_citation
)

from .citation_manager import (
    CitationManager,
    CitationTracker,
    CitationType,
    SourceDocument
)

from .evaluation import (
    RetrievalEvaluator,
    GenerationEvaluator,
    RAGEvaluator,
    RetrievalMetrics,
    GenerationMetrics,
    RAGEvaluationResult
)

from .query_classifier import (
    QueryClassifier,
    QueryRouter,
    PromptSelector,
    QueryType,
    LegalDomain,
    suggest_follow_up_queries
)

__all__ = [
    "LegalTextProcessor",
    "chunk_documents",
    "extract_citations",
    "normalize_citation",
    "CitationManager",
    "CitationTracker",
    "CitationType",
    "SourceDocument",
    "RetrievalEvaluator",
    "GenerationEvaluator",
    "RAGEvaluator",
    "RetrievalMetrics",
    "GenerationMetrics",
    "RAGEvaluationResult",
    "QueryClassifier",
    "QueryRouter",
    "PromptSelector",
    "QueryType",
    "LegalDomain",
    "suggest_follow_up_queries",
]
