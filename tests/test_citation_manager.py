"""
Tests for citation management utilities.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.citation_manager import (
    CitationManager,
    CitationTracker,
    CitationType,
    LegalCitation,
    SourceDocument
)


class TestCitationManager:
    """Test cases for CitationManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = CitationManager()

    def test_parse_article_citation(self):
        """Test parsing article citations."""
        text = "Article L1234-1 du Code du travail"
        citation = self.manager.parse_citation(text)
        assert citation.citation_type == CitationType.ARTICLE_CODE
        assert citation.article == "L1234-1"
        assert citation.code is not None and "travail" in citation.code.lower()
        assert citation.confidence == 1.0

    def test_parse_law_citation(self):
        """Test parsing law citations."""
        text = "loi n° 2008-596 du 25 juin 2008"
        citation = self.manager.parse_citation(text)
        assert citation.citation_type == CitationType.LAW
        assert citation.law_number == "2008-596"
        assert citation.date == "25 juin 2008"

    def test_parse_decree_citation(self):
        """Test parsing decree citations."""
        text = "décret n° 2019-1344 du 12 décembre 2019"
        citation = self.manager.parse_citation(text)
        assert citation.citation_type == CitationType.DECREE
        assert citation.decree_number == "2019-1344"

    def test_normalize_citation(self):
        """Test citation normalization."""
        citation = "Article   L1234-1   du   Code   du   travail"
        normalized = self.manager.normalize_citation(citation)
        assert normalized == "Art. L1234-1 du Code du travail"
        assert "  " not in normalized

    def test_extract_citations_from_text(self):
        """Test extracting multiple citations from text."""
        text = """
        According to Article L1234-1 du Code du travail,
        and loi n° 2008-596 du 25 juin 2008,
        see also Art. L1234-2 du Code du travail.
        """
        citations = self.manager.extract_citations_from_text(text)
        assert len(citations) >= 3

    def test_register_and_get_source(self):
        """Test registering and retrieving source documents."""
        source = SourceDocument(
            doc_id="test_doc_1",
            text="Test text content",
            code="Code du travail",
            article="L1234-1"
        )
        self.manager.register_source(source)

        retrieved = self.manager.get_source("Code du travail", "L1234-1")
        assert retrieved is not None
        assert retrieved.doc_id == "test_doc_1"

    def test_format_citation_for_answer(self):
        """Test formatting citations for answers."""
        citation = LegalCitation(
            citation_type=CitationType.ARTICLE_CODE,
            raw_text="Article L1234-1 du Code du travail",
            normalized="Art. L1234-1 du Code du travail",
            article="L1234-1",
            code="travail"
        )
        formatted = self.manager.format_citation_for_answer(citation)
        assert "Art." in formatted
        assert "L1234-1" in formatted
        assert "Code" in formatted

    def test_verify_citation_in_context(self):
        """Test citation verification."""
        citation = LegalCitation(
            citation_type=CitationType.ARTICLE_CODE,
            raw_text="Article L1234-1 du Code du travail",
            normalized="Art. L1234-1 du Code du travail",
            article="L1234-1",
            code="travail"
        )
        context = "According to Article L1234-1 du Code du travail, the trial period is defined."

        found, confidence = self.manager.verify_citation_in_context(citation, context)
        assert found is True
        assert confidence >= 0.7

    def test_generate_citation_list(self):
        """Test generating formatted citation lists."""
        citations = [
            self.manager.parse_citation("Article L1234-1 du Code du travail"),
            self.manager.parse_citation("loi n° 2008-596"),
        ]
        citation_list = self.manager.generate_citation_list(citations)
        assert "[1]" in citation_list
        assert "[2]" in citation_list


class TestCitationTracker:
    """Test cases for CitationTracker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = CitationTracker()

    def test_track_citation(self):
        """Test tracking citations."""
        citation = LegalCitation(
            citation_type=CitationType.ARTICLE_CODE,
            raw_text="Article L1234-1 du Code du travail",
            normalized="Art. L1234-1 du Code du travail",
            article="L1234-1",
            code="travail"
        )
        self.tracker.track_citation(citation, "doc1", 0.95)
        assert len(self.tracker.citations) == 1
        assert "doc1" in self.tracker.sources_used

    def test_get_top_sources(self):
        """Test getting top sources."""
        citation1 = LegalCitation(
            citation_type=CitationType.ARTICLE_CODE,
            raw_text="Article L1234-1 du Code du travail",
            normalized="Art. L1234-1 du Code du travail",
            article="L1234-1",
            code="travail"
        )
        citation2 = LegalCitation(
            citation_type=CitationType.ARTICLE_CODE,
            raw_text="Article L1234-2 du Code du travail",
            normalized="Art. L1234-2 du Code du travail",
            article="L1234-2",
            code="travail"
        )

        self.tracker.track_citation(citation1, "doc1", 0.9)
        self.tracker.track_citation(citation1, "doc1", 0.8)
        self.tracker.track_citation(citation2, "doc2", 0.7)

        top_sources = self.tracker.get_top_sources(limit=2)
        assert len(top_sources) == 2
        assert top_sources[0] == ("doc1", 2)

    def test_get_citation_report(self):
        """Test generating citation reports."""
        citation = LegalCitation(
            citation_type=CitationType.ARTICLE_CODE,
            raw_text="Article L1234-1 du Code du travail",
            normalized="Art. L1234-1 du Code du travail",
            article="L1234-1",
            code="travail"
        )
        self.tracker.track_citation(citation, "doc1", 0.9)

        report = self.tracker.get_citation_report()
        assert "total_citations" in report
        assert "unique_sources" in report
        assert report["total_citations"] == 1

    def test_reset(self):
        """Test tracker reset."""
        citation = LegalCitation(
            citation_type=CitationType.ARTICLE_CODE,
            raw_text="Test",
            normalized="Test",
        )
        self.tracker.track_citation(citation, "doc1", 0.9)
        self.tracker.reset()
        assert len(self.tracker.citations) == 0
        assert len(self.tracker.sources_used) == 0
