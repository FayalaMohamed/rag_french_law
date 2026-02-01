"""
Tests for text processing utilities.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_processing import (
    LegalTextProcessor,
    Chunk,
    chunk_documents,
    extract_citations,
    normalize_citation
)


class TestLegalTextProcessor:
    """Test cases for LegalTextProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = LegalTextProcessor(chunk_size=500, chunk_overlap=100)

    def test_clean_text(self):
        """Test text cleaning functionality."""
        text = "This   is  a   test.\n\nWith   extra   spaces."
        cleaned = self.processor.clean_text(text)
        assert "  " not in cleaned
        assert "\n\n" not in cleaned
        assert "This is a test" in cleaned
        assert "With extra spaces" in cleaned

    def test_normalize_french_text(self):
        """Test French text normalization."""
        text = "œuf, æther, « guillemets », …"
        normalized = self.processor.normalize_french_text(text)
        assert "œ" not in normalized
        assert "æ" not in normalized
        assert "«" not in normalized
        assert "»" not in normalized

    def test_extract_article_references(self):
        """Test article reference extraction."""
        text = "Selon l'Article L1234-1 du Code du travail et l'Art. 1101 du Code civil."
        references = self.processor.extract_article_references(text)
        assert len(references) == 2
        assert references[0]["article"] == "L1234-1"
        assert "travail" in references[0]["code"].lower()
        assert references[1]["article"] == "1101"
        assert "civil" in references[1]["code"].lower()

    def test_split_into_sentences(self):
        """Test sentence splitting."""
        text = "Première phrase. Deuxième phrase! Troisième phrase?"
        sentences = self.processor.split_into_sentences(text)
        assert len(sentences) == 3
        assert "Première phrase" in sentences[0]
        assert "Deuxième phrase" in sentences[1]
        assert "Troisième phrase" in sentences[2]

    def test_split_into_paragraphs(self):
        """Test paragraph splitting."""
        text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        paragraphs = self.processor.split_into_paragraphs(text)
        assert len(paragraphs) == 3

    def test_create_semantic_chunks(self):
        """Test semantic chunk creation."""
        text = "This is a test document. " * 50
        chunks = self.processor.create_semantic_chunks(text)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.text) <= 500
            assert chunk.chunk_index >= 0

    def test_create_overlapping_chunks(self):
        """Test overlapping chunk creation."""
        text = "This is a test document. " * 100
        chunks = self.processor.create_overlapping_chunks(text)
        assert len(chunks) > 0
        assert len(chunks) >= len(self.processor.create_semantic_chunks(text))

    def test_extract_metadata_from_text(self):
        """Test metadata extraction."""
        text = "Article L1234-1 du Code du travail mentionne également l'Article 1101 du Code civil."
        metadata = self.processor.extract_metadata_from_text(text)
        assert "article_references" in metadata
        assert len(metadata["article_references"]) == 2
        assert "codes_mentioned" in metadata
        assert "travail" in metadata["codes_mentioned"]
        assert "civil" in metadata["codes_mentioned"]
        assert metadata["text_length"] == len(text)


class TestChunkDocuments:
    """Test cases for chunk_documents function."""

    def test_chunk_documents_basic(self):
        """Test basic document chunking."""
        documents = [
            {"id": "doc1", "text": "This is test document 1. " * 50},
            {"id": "doc2", "text": "This is test document 2. " * 50},
        ]
        chunked = chunk_documents(documents, chunk_size=500, chunk_overlap=50)
        assert len(chunked) > 0
        assert all("text" in doc for doc in chunked)
        assert all("chunk_index" in doc for doc in chunked)
        assert all("doc_id" in doc for doc in chunked)

    def test_chunk_documents_empty_text(self):
        """Test handling of empty documents."""
        documents = [
            {"id": "doc1", "text": ""},
            {"id": "doc2", "text": "Valid text"},
        ]
        chunked = chunk_documents(documents)
        assert len(chunked) == 0


class TestCitationExtraction:
    """Test cases for citation extraction."""

    def test_extract_citations(self):
        """Test citation extraction from text."""
        text = "According to Article L1234-1 of the Labor Code and Art. 1101 of the Civil Code."
        citations = extract_citations(text)
        assert len(citations) > 0

    def test_normalize_citation(self):
        """Test citation normalization."""
        citation = "Article L1234-1 du Code du travail"
        normalized = normalize_citation(citation)
        assert normalized == "Art. L1234-1 du Code du travail"
