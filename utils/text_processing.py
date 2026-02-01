"""
Text processing utilities for French legal documents.
Provides functions for cleaning, normalizing, and chunking legal text.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a chunk of text with metadata."""
    text: str
    start_idx: int
    end_idx: int
    chunk_index: int
    metadata: Dict = None


class LegalTextProcessor:
    """Processor for French legal text documents."""

    ARTICLE_PATTERN = re.compile(
        r"(?:Article|Art\.?)\s*([A-Z0-9\-]+)\s+(?:du|de la|des)\s+Code\s+(?:du\s+)?([a-zA-ZÀ-ÿ]+)",
        re.IGNORECASE
    )

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text processor.

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize French legal text.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        text = text.strip()
        text = re.sub(r'\s*([.,;:!?])\s*', r'\1', text)
        text = re.sub(r'\s*-\s*', '-', text)
        return text

    def normalize_french_text(self, text: str) -> str:
        """
        Normalize French text for consistent processing.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        replacements = {
            'œ': 'oe',
            'æ': 'ae',
            '«': '"',
            '»': '"',
            '…': '...',
            '–': '-',
            '—': '-',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def extract_article_references(self, text: str) -> List[Dict[str, str]]:
        """
        Extract article references from legal text.

        Args:
            text: Legal text containing article references

        Returns:
            List of dictionaries with article and code information
        """
        references = []
        for match in self.ARTICLE_PATTERN.finditer(text):
            article = match.group(1).strip()
            code = match.group(2).strip()
            if article and code:
                references.append({
                    "article": article,
                    "code": code,
                    "raw": match.group(0)
                })
        return references

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, handling French punctuation.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]

    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.

        Args:
            text: Text to split

        Returns:
            List of paragraphs
        """
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]

    def create_semantic_chunks(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        min_chunk_size: int = 100
    ) -> List[Chunk]:
        """
        Create semantically coherent chunks from legal text.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            min_chunk_size: Minimum chunk size

        Returns:
            List of Chunk objects
        """
        text = self.clean_text(text)
        text = self.normalize_french_text(text)

        paragraphs = self.split_into_paragraphs(text)
        chunks = []
        current_chunk = ""
        current_idx = 0
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current_chunk) + len(paragraph) < self.chunk_size:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(Chunk(
                        text=current_chunk[:self.chunk_size],
                        start_idx=current_idx,
                        end_idx=current_idx + len(current_chunk[:self.chunk_size]),
                        chunk_index=chunk_index,
                        metadata=metadata.copy() if metadata else {}
                    ))
                    chunk_index += 1
                    current_idx += len(current_chunk[:self.chunk_size])

                if len(paragraph) > self.chunk_size:
                    sentences = self.split_into_sentences(paragraph)
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) < self.chunk_size:
                            current_chunk += " " + sentence if current_chunk else sentence
                        else:
                            if current_chunk:
                                chunks.append(Chunk(
                                    text=current_chunk[:self.chunk_size],
                                    start_idx=current_idx,
                                    end_idx=current_idx + len(current_chunk[:self.chunk_size]),
                                    chunk_index=chunk_index,
                                    metadata=metadata.copy() if metadata else {}
                                ))
                                chunk_index += 1
                                current_idx += len(current_chunk[:self.chunk_size])

                            current_chunk = sentence
                else:
                    current_chunk = paragraph

        if current_chunk and len(current_chunk) >= min_chunk_size:
            chunks.append(Chunk(
                text=current_chunk[:self.chunk_size],
                start_idx=current_idx,
                end_idx=current_idx + len(current_chunk[:self.chunk_size]),
                chunk_index=chunk_index,
                metadata=metadata.copy() if metadata else {}
            ))

        return chunks

    def create_overlapping_chunks(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        min_chunk_size: int = 100
    ) -> List[Chunk]:
        """
        Create overlapping chunks for better context preservation.

        Args:
            text: Text to chunk
            metadata: Optional metadata
            min_chunk_size: Minimum chunk size

        Returns:
            List of Chunk objects
        """
        non_overlapping = self.create_semantic_chunks(text, metadata, min_chunk_size)
        if len(non_overlapping) < 2:
            return non_overlapping

        chunks = []
        for i, chunk in enumerate(non_overlapping):
            chunks.append(chunk)

            if i < len(non_overlapping) - 1:
                next_chunk = non_overlapping[i + 1]
                overlap_text = chunk.text[-self.chunk_overlap:] + " " + next_chunk.text[:self.chunk_overlap]

                if len(overlap_text) >= min_chunk_size:
                    overlap_chunk = Chunk(
                        text=overlap_text,
                        start_idx=chunk.end_idx - self.chunk_overlap,
                        end_idx=next_chunk.start_idx + self.chunk_overlap,
                        chunk_index=len(chunks),
                        metadata={(metadata.copy() if metadata else {})}
                    )
                    chunks.append(overlap_chunk)

        return chunks

    def extract_metadata_from_text(self, text: str) -> Dict:
        """
        Extract metadata from legal text.

        Args:
            text: Legal text

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            "article_references": [],
            "codes_mentioned": set(),
            "has_enumerations": bool(re.search(r'\d+\.', text)),
            "has_references": bool(re.search(r'(?:voir|consulter|art\.?|article)', text, re.IGNORECASE)),
            "text_length": len(text),
            "sentence_count": len(self.split_into_sentences(text)),
            "paragraph_count": len(self.split_into_paragraphs(text)),
        }

        references = self.extract_article_references(text)
        metadata["article_references"] = references
        for ref in references:
            code = ref["code"]
            if code.startswith("du "):
                code = code[3:]
            elif code.startswith("de la "):
                code = code[6:]
            elif code.startswith("des "):
                code = code[4:]
            metadata["codes_mentioned"].add(code)

        metadata["codes_mentioned"] = list(metadata["codes_mentioned"])

        return metadata


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    text_field: str = "text",
    metadata_fields: List[str] = None
) -> List[Dict]:
    """
    Chunk a list of documents.

    Args:
        documents: List of document dictionaries
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        text_field: Field name for text content
        metadata_fields: Fields to include in chunk metadata

    Returns:
        List of chunked documents
    """
    processor = LegalTextProcessor(chunk_size, chunk_overlap)
    chunked_documents = []

    for doc_idx, doc in enumerate(documents):
        text = doc.get(text_field, "")
        if not text:
            continue

        base_metadata = {"doc_id": doc.get("id", f"doc_{doc_idx}")}
        if metadata_fields:
            for field in metadata_fields:
                if field in doc:
                    base_metadata[field] = doc[field]

        chunks = processor.create_overlapping_chunks(text, base_metadata)

        for chunk in chunks:
            chunk_doc = {
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "doc_id": base_metadata.get("doc_id"),
                "metadata": {
                    **base_metadata,
                    **chunk.metadata,
                    "start_idx": chunk.start_idx,
                    "end_idx": chunk.end_idx,
                    "original_length": len(text)
                }
            }
            chunked_documents.append(chunk_doc)

    return chunked_documents


def extract_citations(text: str) -> List[Dict[str, str]]:
    """
    Extract all legal citations from text.

    Args:
        text: Text containing citations

    Returns:
        List of citation dictionaries
    """
    citations = []

    patterns = [
        (r"Article\s+([A-Z0-9\-]+)\s+du\s+Code\s+([A-Za-zÀ-ÿ]+)", "article_code"),
        (r"Art\.?\s*([A-Z0-9\-]+)\s+(?:du|de la|des)?\s*Code\s+([A-Za-zÀ-ÿ]+)", "article_code_short"),
        (r"Article\s+([A-Z0-9\-]+)\s+of\s+the\s+([A-Za-zÀ-ÿ]+)\s+Code", "article_code_english"),
        (r"Art\.?\s*([A-Z0-9\-]+)\s+of\s+the\s+([A-Za-zÀ-ÿ]+)\s+Code", "article_code_short_english"),
        (r"Code\s+([A-Za-zÀ-ÿ]+)\s+article\s+([A-Z0-9\-]+)", "code_article"),
        (r"loi\s+n°\s*([0-9\-]+)", "law_number"),
        (r"décret\s+n°\s*([0-9\-]+)", "decree_number"),
    ]

    for pattern, citation_type in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            citations.append({
                "type": citation_type,
                "raw": match.group(0),
                "extracted": match.groups(),
                "start": match.start(),
                "end": match.end()
            })

    return citations


def normalize_citation(citation: str) -> str:
    """
    Normalize a legal citation to standard format.

    Args:
        citation: Raw citation string

    Returns:
        Normalized citation string
    """
    citation = citation.strip()
    citation = re.sub(r'\s+', ' ', citation)
    citation = re.sub(r'^Article\s+', 'Art. ', citation, flags=re.IGNORECASE)
    return citation


if __name__ == "__main__":
    processor = LegalTextProcessor(chunk_size=500, chunk_overlap=100)

    sample_text = """
    Article L1234-1 du Code du travail

    La période d'essai permet aux parties d'évaluer les conditions de travail et de vérifier la conformité des attentes respectives.

    Article L1234-2 du Code du travail

    La durée de la période d'essai ne peut être indéfinie. Elle est ditentukan selon les dispositions suivantes.
    """

    print("Original text:")
    print(sample_text)
    print("\n" + "="*50)

    print("\nCleaned text:")
    print(processor.clean_text(sample_text))
    print("\n" + "="*50)

    print("\nExtracted references:")
    print(processor.extract_article_references(sample_text))
    print("\n" + "="*50)

    print("\nChunks:")
    chunks = processor.create_semantic_chunks(sample_text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk.text[:100]}...")
