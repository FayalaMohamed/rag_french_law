"""
Citation and source management utilities for French legal RAG.
Provides functionality for tracking, formatting, and verifying legal citations.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import json


class CitationType(Enum):
    """Types of legal citations."""
    ARTICLE_CODE = "article_code"
    LAW = "law"
    DECREE = "decree"
    JURISPRUDENCE = "jurisprudence"
    EUROPEAN = "european"
    CONSTITUTIONAL = "constitutional"


@dataclass
class LegalCitation:
    """Represents a legal citation."""
    citation_type: CitationType
    raw_text: str
    normalized: str
    article: Optional[str] = None
    code: Optional[str] = None
    law_number: Optional[str] = None
    decree_number: Optional[str] = None
    jurisdiction: Optional[str] = None
    date: Optional[str] = None
    page: Optional[str] = None
    source_document: Optional[str] = None
    confidence: float = 1.0


@dataclass
class SourceDocument:
    """Represents a source legal document."""
    doc_id: str
    text: str
    code: str
    article: str
    section: Optional[str] = None
    chapter: Optional[str] = None
    title: Optional[str] = None
    book: Optional[str] = None
    raw_metadata: Dict = field(default_factory=dict)
    citation_count: int = 0
    last_accessed: str = ""


class CitationManager:
    """Manages citations and source tracking for the RAG system."""

    ARTICLE_PATTERN = re.compile(
        r"(?:Article|Art\.?)\s*(?:No\.?\s*)?([A-Z0-9\-/]+)\s+(?:du|de la|des)\s+Code\s+du\s+([a-zA-ZÀ-ÿ]+)",
        re.IGNORECASE
    )

    LAW_PATTERN = re.compile(
        r"loi\s+(?:n°|numéro)?\s*([0-9\-/]+)\s*(?:du\s+)?(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4})?",
        re.IGNORECASE
    )

    DECREE_PATTERN = re.compile(
        r"décret\s+(?:n°|numéro)?\s*([0-9\-/]+)\s*(?:du\s+)?([0-9/\-]+)?",
        re.IGNORECASE
    )

    def __init__(self):
        """Initialize the citation manager."""
        self.citations: List[LegalCitation] = []
        self.source_documents: Dict[str, SourceDocument] = {}
        self.citation_index: Dict[str, List[int]] = {}

    def parse_citation(self, text: str) -> LegalCitation:
        """
        Parse a citation string into a LegalCitation object.

        Args:
            text: Raw citation text

        Returns:
            LegalCitation object
        """
        text = text.strip()
        normalized = self.normalize_citation(text)

        for pattern, ctype in [
            (self.ARTICLE_PATTERN, CitationType.ARTICLE_CODE),
            (self.LAW_PATTERN, CitationType.LAW),
            (self.DECREE_PATTERN, CitationType.DECREE),
        ]:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                citation = LegalCitation(
                    citation_type=ctype,
                    raw_text=text,
                    normalized=normalized,
                )

                if ctype == CitationType.ARTICLE_CODE:
                    citation.article = groups[0] if groups else None
                    citation.code = groups[1] if groups else None
                elif ctype == CitationType.LAW:
                    citation.law_number = groups[0] if groups else None
                    citation.date = groups[1] if groups else None
                elif ctype == CitationType.DECREE:
                    citation.decree_number = groups[0] if groups else None
                    citation.date = groups[1] if groups else None

                return citation

        return LegalCitation(
            citation_type=CitationType.ARTICLE_CODE,
            raw_text=text,
            normalized=normalized,
            confidence=0.5
        )

    def normalize_citation(self, citation: str) -> str:
        """
        Normalize a citation to standard format.

        Args:
            citation: Raw citation string

        Returns:
            Normalized citation string
        """
        citation = citation.strip()
        citation = re.sub(r'\s+', ' ', citation)
        citation = re.sub(r'^Article\s+', 'Art. ', citation, flags=re.IGNORECASE)
        citation = re.sub(r'^Art\.?\s*', 'Art. ', citation, flags=re.IGNORECASE)
        citation = re.sub(r'\s*du\s+Code\s+', ' du Code ', citation, flags=re.IGNORECASE)
        citation = re.sub(r'\s*de la\s+Code\s+', ' de la Code ', citation, flags=re.IGNORECASE)
        return citation

    def extract_citations_from_text(self, text: str) -> List[LegalCitation]:
        """
        Extract all citations from a text.

        Args:
            text: Text to search for citations

        Returns:
            List of LegalCitation objects
        """
        citations = []

        for pattern, ctype in [
            (self.ARTICLE_PATTERN, CitationType.ARTICLE_CODE),
            (self.LAW_PATTERN, CitationType.LAW),
            (self.DECREE_PATTERN, CitationType.DECREE),
        ]:
            for match in pattern.finditer(text):
                match_text = match.group(0)
                citation = self.parse_citation(match_text)
                citations.append(citation)

        return citations

    def register_source(self, source: SourceDocument):
        """
        Register a source document.

        Args:
            source: SourceDocument to register
        """
        key = f"{source.code}_{source.article}"
        self.source_documents[key] = source
        self.citation_index[key] = self.citation_index.get(key, [])
        self.citation_index[key].append(source.citation_count)
        source.citation_count += 1

    def get_source(self, code: str, article: str) -> Optional[SourceDocument]:
        """
        Get a source document by code and article.

        Args:
            code: Legal code name
            article: Article number

        Returns:
            SourceDocument if found, None otherwise
        """
        key = f"{code}_{article}"
        return self.source_documents.get(key)

    def format_citation_for_answer(
        self,
        citation: LegalCitation,
        include_context: bool = True
    ) -> str:
        """
        Format a citation for inclusion in an answer.

        Args:
            citation: Citation to format
            include_context: Whether to include source document context

        Returns:
            Formatted citation string
        """
        if citation.citation_type == CitationType.ARTICLE_CODE:
            formatted = f"Art. {citation.article} du Code {citation.code}"
            if include_context and citation.source_document:
                formatted += f" ({citation.source_document})"
        elif citation.citation_type == CitationType.LAW:
            formatted = f"Loi n°{citation.law_number}"
            if citation.date:
                formatted += f" du {citation.date}"
        elif citation.citation_type == CitationType.DECREE:
            formatted = f"Décret n°{citation.decree_number}"
            if citation.date:
                formatted += f" du {citation.date}"
        else:
            formatted = citation.normalized

        return formatted

    def verify_citation_in_context(
        self,
        citation: LegalCitation,
        context: str
    ) -> Tuple[bool, float]:
        """
        Verify that a citation exists in the provided context.

        Args:
            citation: Citation to verify
            context: Context text to search in

        Returns:
            Tuple of (found, confidence_score)
        """
        search_pattern = re.escape(citation.raw_text)
        match = re.search(search_pattern, context, re.IGNORECASE)

        if match:
            return True, 1.0

        normalized_pattern = re.escape(citation.normalized)
        normalized_match = re.search(normalized_pattern, context, re.IGNORECASE)

        if normalized_match:
            return True, 0.9

        if citation.citation_type == CitationType.ARTICLE_CODE and citation.article and citation.code:
            partial_pattern = rf"{re.escape(citation.article)}.*(?:du|de la)\s*Code\s*{re.escape(citation.code)}"
            partial_match = re.search(partial_pattern, context, re.IGNORECASE | re.DOTALL)
            if partial_match:
                return True, 0.7

        return False, 0.0

    def generate_citation_list(self, citations: List[LegalCitation]) -> str:
        """
        Generate a formatted list of citations.

        Args:
            citations: List of citations to format

        Returns:
            Formatted citation list string
        """
        if not citations:
            return ""

        unique_citations = []
        seen = set()

        for citation in citations:
            key = citation.normalized
            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)

        formatted = []
        for i, citation in enumerate(unique_citations, 1):
            formatted.append(f"[{i}] {self.format_citation_for_answer(citation)}")

        return "\n".join(formatted)


class CitationTracker:
    """Tracks citations used in RAG responses."""

    def __init__(self):
        """Initialize the citation tracker."""
        self.citations: List[Dict] = []
        self.sources_used: Dict[str, int] = {}

    def track_citation(
        self,
        citation: LegalCitation,
        source_doc_id: str,
        relevance_score: float
    ):
        """
        Track a citation used in a response.

        Args:
            citation: Citation that was used
            source_doc_id: ID of the source document
            relevance_score: Relevance score from retrieval
        """
        citation_entry = {
            "citation": citation.normalized,
            "source_doc_id": source_doc_id,
            "relevance_score": relevance_score,
            "citation_type": citation.citation_type.value,
            "article": citation.article,
            "code": citation.code,
        }
        self.citations.append(citation_entry)

        if source_doc_id in self.sources_used:
            self.sources_used[source_doc_id] += 1
        else:
            self.sources_used[source_doc_id] = 1

    def get_top_sources(self, limit: int = 5) -> List[Tuple[str, int]]:
        """
        Get the most frequently used sources.

        Args:
            limit: Maximum number of sources to return

        Returns:
            List of (source_id, count) tuples
        """
        sorted_sources = sorted(
            self.sources_used.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_sources[:limit]

    def get_citation_report(self) -> Dict:
        """
        Generate a report of citations used.

        Returns:
            Dictionary with citation statistics
        """
        return {
            "total_citations": len(self.citations),
            "unique_sources": len(self.sources_used),
            "citations_by_type": self._count_by_type(),
            "top_sources": self.get_top_sources(10),
        }

    def _count_by_type(self) -> Dict[str, int]:
        """Count citations by type."""
        counts = {}
        for citation in self.citations:
            ctype = citation["citation_type"]
            counts[ctype] = counts.get(ctype, 0) + 1
        return counts

    def reset(self):
        """Reset the tracker."""
        self.citations = []
        self.sources_used = {}


if __name__ == "__main__":
    manager = CitationManager()

    sample_text = """
    Selon l'Article L1234-1 du Code du travail, la période d'essai permet aux parties
    d'évaluer les conditions de travail. Par ailleurs, la loi n° 2008-596 du 25 juin 2008
    a modifié ces dispositions. Voir également l'Art. L1234-2 du Code du travail
    concernant la durée de la période d'essai.
    """

    print("Extracted citations:")
    citations = manager.extract_citations_from_text(sample_text)
    for citation in citations:
        print(f"- Type: {citation.citation_type.value}")
        print(f"  Raw: {citation.raw_text}")
        print(f"  Normalized: {citation.normalized}")
        if citation.article:
            print(f"  Article: {citation.article}")
        if citation.code:
            print(f"  Code: {citation.code}")
        print()

    print("\nFormatted citation list:")
    print(manager.generate_citation_list(citations))
