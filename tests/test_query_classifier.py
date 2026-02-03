"""
Tests for query classification and routing.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.query_classifier import (
    QueryClassifier,
    QueryRouter,
    PromptSelector,
    QueryType,
    LegalDomain,
    suggest_follow_up_queries
)


class TestQueryClassifier(unittest.TestCase):
    """Test cases for QueryClassifier."""

    def setUp(self):
        """Set up test fixtures."""
        self.classifier = QueryClassifier()

    def test_classify_definition_query(self):
        """Test classification of definition queries."""
        query = "Qu'est-ce que la période d'essai?"
        query_type, confidence = self.classifier.classify_query_type(query)
        assert query_type == QueryType.DEFINITION
        assert confidence > 0.5

    def test_classify_procedural_query(self):
        """Test classification of procedural queries."""
        query = "Comment rompre un contrat de travail?"
        query_type, confidence = self.classifier.classify_query_type(query)
        assert query_type == QueryType.PROCEDURAL
        assert confidence > 0.5

    def test_classify_conditional_query(self):
        """Test classification of conditional queries."""
        query = "Quelles sont les conditions pour demander un congé?"
        query_type, confidence = self.classifier.classify_query_type(query)
        assert query_type == QueryType.CONDITIONAL
        assert confidence > 0.5

    def test_classify_temporal_query(self):
        """Test classification of temporal queries."""
        query = "Quel est le délai de préavis?"
        query_type, confidence = self.classifier.classify_query_type(query)
        assert query_type == QueryType.TEMPORAL
        assert confidence > 0.5

    def test_classify_citation_query(self):
        """Test classification of citation lookup queries."""
        query = "Article L1234-1 du Code du travail"
        query_type, confidence = self.classifier.classify_query_type(query)
        assert query_type == QueryType.CITATION_LOOKUP

    def test_classify_labor_law_domain(self):
        """Test domain classification for labor law."""
        query = "Quelles sont les conditions de la période d'essai?"
        domain, confidence = self.classifier.classify_domain(query)
        assert domain == LegalDomain.LABOR_LAW
        assert confidence > 0.5

    def test_classify_civil_law_domain(self):
        """Test domain classification for civil law."""
        query = "Qu'est-ce qu'un contrat de location?"
        domain, confidence = self.classifier.classify_domain(query)
        assert domain == LegalDomain.CIVIL_LAW

    def test_classify_general_domain(self):
        """Test default domain classification."""
        query = "What is the meaning of life?"
        domain, confidence = self.classifier.classify_domain(query)
        assert domain == LegalDomain.GENERAL

    def test_classify_complete(self):
        """Test complete query classification."""
        query = "Comment rédiger une rupture conventionnelle?"
        result = self.classifier.classify(query)

        assert "query_type" in result
        assert "domain" in result
        assert "is_urgent" in result
        assert "complexity" in result
        assert result["query_type"] == QueryType.PROCEDURAL.value
        assert result["domain"] == LegalDomain.LABOR_LAW.value


class TestQueryRouter(unittest.TestCase):
    """Test cases for QueryRouter."""

    def setUp(self):
        """Set up test fixtures."""
        self.router = QueryRouter()

    def test_route_definition_query(self):
        """Test routing for definition queries."""
        result = self.router.route("Qu'est-ce que la période d'essai?")
        assert result["retrieval_params"]["k"] == 3
        assert result["retrieval_params"]["use_decomposition"] is False

    def test_route_procedural_query(self):
        """Test routing for procedural queries."""
        result = self.router.route("Comment rompre un contrat de travail?")
        assert result["retrieval_params"]["k"] >= 5
        assert result["retrieval_params"]["use_decomposition"] is True

    def test_route_citation_query(self):
        """Test routing for citation lookup queries."""
        result = self.router.route("Article L1234-1 du Code du travail")
        assert result["retrieval_params"]["k"] == 3
        assert result["retrieval_params"]["use_hyde"] is False

    def test_route_complex_query(self):
        """Test routing for complex queries."""
        result = self.router.route("Quelles sont les règles concernant le contrat de travail, la période d'essai, et les conditions de rupture?")
        assert result["retrieval_params"]["k"] >= 5
        assert result["retrieval_params"]["use_decomposition"] is True

    def test_route_urgent_query(self):
        """Test routing for urgent queries."""
        result = self.router.route("URGENT: Comment rompre un contrat immédiatement?")
        assert result["classification"]["is_urgent"] is True
        assert result["retrieval_params"]["k"] <= 5


class TestPromptSelector(unittest.TestCase):
    """Test cases for PromptSelector."""

    def setUp(self):
        """Set up test fixtures."""
        self.selector = PromptSelector()

    def test_get_prompt_for_definition(self):
        """Test getting prompt for definition queries."""
        prompt_name = self.selector.get_prompt_name(QueryType.DEFINITION)
        assert prompt_name == "definition_prompt"

    def test_get_prompt_for_procedural(self):
        """Test getting prompt for procedural queries."""
        prompt_name = self.selector.get_prompt_name(QueryType.PROCEDURAL)
        assert prompt_name == "procedural_prompt"

    def test_get_prompt_for_general(self):
        """Test getting prompt for general queries."""
        prompt_name = self.selector.get_prompt_name(QueryType.GENERAL)
        assert prompt_name == "general_prompt"


class TestFollowUpSuggestions(unittest.TestCase):
    """Test cases for follow-up query suggestions."""

    def test_suggest_for_definition(self):
        """Test suggestions for definition queries."""
        suggestions = suggest_follow_up_queries(
            QueryType.DEFINITION,
            "Qu'est-ce que la période d'essai?"
        )
        assert len(suggestions) > 0
        assert any("exception" in s.lower() for s in suggestions)

    def test_suggest_for_procedural(self):
        """Test suggestions for procedural queries."""
        suggestions = suggest_follow_up_queries(
            QueryType.PROCEDURAL,
            "Comment rédiger un contrat?"
        )
        assert len(suggestions) > 0

    def test_suggest_for_temporal(self):
        """Test suggestions for temporal queries."""
        suggestions = suggest_follow_up_queries(
            QueryType.TEMPORAL,
            "Quel est le délai?"
        )
        assert len(suggestions) > 0
