# AGENTS.md - French Legal RAG System

This document provides guidelines for agentic coding agents working on this French Legal RAG (Retrieval-Augmented Generation) codebase.

## Project Overview

A RAG system for French legal document question answering using:
- **Language**: Python 3.10-3.12
- **Vector Store**: FAISS for document retrieval
- **Embeddings**: multilingual-e5-small (multilingual E5 model optimized for French)
- **LLM**: Ollama (ministral-3:3b-instruct or similar)
- **Framework**: LangChain

## Build/Lint/Test Commands

### Installation
```bash
# Create conda environment
conda env create -f environment.yml
conda activate rag-env

# Or install directly with pip
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests
pytest

# Run single test file
pytest tests/test_text_processing.py

# Run single test
pytest tests/test_text_processing.py::TestLegalTextProcessor::test_clean_text

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=.

# Run matching pattern
pytest -k "test_parse"
```

### Running the Application
```bash
# Build the RAG index
python main.py --build

# Limit articles for faster testing
python main.py --build --limit 100

# Run test queries
python main.py --test

# Interactive query mode
python main.py --interactive
```

### Running Benchmarks
```bash
# Generate test queries
python benchmark/generate_test_queries.py --output benchmark_results/test_queries_generated.json

# Compare retrieval strategies
python benchmark/compare_retrieval_strategies.py --run-comparison

# Compare embedding models
python benchmark/compare_embeddings.py --full-evaluation

# Benchmark FAISS indices
python benchmark/benchmark_faiss_indices.py --dataset_size 1000 --queries 100

# Benchmark text processing
python benchmark/benchmark_text_processing.py --limit 500
```

### Environment Setup
```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
- `OPENAI_API_KEY`: OpenAI API key (required for some LLM operations)
- `HF_TOKEN`: HuggingFace token (optional, for downloading models)
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)

## Code Style Guidelines

### Import Organization
Order imports in this sequence with blank lines between groups:
1. Standard library imports (`os`, `sys`, `json`, `re`, `typing`)
2. Third-party imports (`langchain`, `faiss`, `pytest`, `datasets`)
3. Local application imports (relative imports with `.`)

```python
import os
import sys
import json
from typing import List, Dict, Optional

import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from models.embeddings import EmbeddingModel
from chains.rag_chain import RAGPipeline
import config
```

### Naming Conventions
- **Classes**: PascalCase (`RAGPipeline`, `LegalTextProcessor`)
- **Functions/Variables**: snake_case (`load_data`, `embedding_dim`)
- **Constants**: UPPER_SNAKE_CASE (`DATASET_NAME`, `TOP_K_RETRIEVAL`)
- **Private Methods**: Leading underscore (`_internal_method`)
- **Type Variables**: PascalCase when used in generics

### Type Hints
Always use type hints for function signatures. Import from `typing`:
```python
from typing import List, Dict, Tuple, Optional, Union

def process_documents(
    documents: List[str],
    metadata: Optional[Dict] = None,
    batch_size: int = 256
) -> Tuple[np.ndarray, List[Dict]]:
```

### Docstring Format
Use Google-style docstrings for all public functions and classes:
```python
def search_by_text(
    self,
    query_text: str,
    embedding_model,
    k: int = 5
) -> Tuple[List[str], List[float], List[Dict]]:
    """
    Search for documents by text query.

    Args:
        query_text: The text query to search for
        embedding_model: Model to generate query embeddings
        k: Number of results to return

    Returns:
        Tuple of (documents, distances, metadatas)
    """
```

### Error Handling
- Use explicit try/except blocks for external operations
- Raise `ValueError` for invalid arguments
- Provide meaningful error messages
- Use `getattr(response, "content", getattr(response, "text", ""))` for LLM response handling

### Code Structure
- Keep functions focused (single responsibility)
- Maximum ~100 lines per function
- Use dataclasses for simple data structures
- Factory functions for object creation (`create_rag_pipeline()`)

### File Organization
```
RAG/
├── main.py              # Entry point with CLI
├── config.py            # Configuration constants
├── benchmark/           # Benchmark scripts and results
│   ├── benchmark_faiss_indices.py
│   ├── benchmark_text_processing.py
│   ├── compare_embeddings.py
│   ├── compare_retrieval_strategies.py
│   ├── generate_test_queries.py
│   └── results/         # Benchmark outputs (JSON, PNG, .faiss)
├── chains/              # RAG chain implementations
│   ├── llm_chain.py    # LLM wrapper with HyDE and decomposition
│   └── rag_chain.py    # Main RAG pipeline
├── data/                # Data loading and storage
│   ├── loader.py       # Dataset loading from HuggingFace
│   └── vector_store.py # FAISS vector store wrapper
├── models/              # Model wrappers
│   └── embeddings.py   # Embedding model wrapper
├── prompts/             # Prompt templates
│   └── prompts.py
├── tests/               # Test files
│   ├── test_text_processing.py
│   ├── test_evaluation.py
│   └── test_query_classifier.py
└── utils/               # Utility functions
    ├── text_processing.py     # Text chunking and cleaning
    ├── query_classifier.py    # Query type classification
    └── evaluation.py          # RAG evaluation metrics
```

### Testing Conventions
- Use `pytest` framework
- Test class naming: `Test{ClassName}`
- Setup method: `setup_method(self)` for fixtures
- Test naming: `test_{method_name}_{scenario}`
- Import path: `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`

```python
class TestTextProcessor:
    def setup_method(self):
        self.processor = LegalTextProcessor()

    def test_clean_text_removes_extra_whitespace(self):
        result = self.processor.clean_text("  text  with  spaces  ")
        assert result == "text with spaces"
```

### Configuration
- Store constants in `config.py`
- Use `os.getenv()` with defaults for environment variables
- Constants in UPPER_SNAKE_CASE

### Common Patterns
- Use `os.makedirs(path, exist_ok=True)` for directory creation
- Use `json.dump(data, f, ensure_ascii=False, indent=2)` for JSON with French text
- Use `re.IGNORECASE` for French text patterns
- Check CUDA availability: `torch.cuda.is_available()`
- Progress bars: `tqdm` for long operations
- Use `os.path.join()` for cross-platform path construction

### French Text Handling
- Always use UTF-8 encoding (`encoding='utf-8'`)
- Normalize special characters (œ, æ, «», …)
- Use French punctuation patterns for sentence splitting
- Handle accented characters in regex patterns

## RAG Pipeline Architecture

The system uses a modular RAG pipeline with the following flow:

1. **Query Input** → QueryClassifier (determines query type: factual/analytical/multi-hop)
2. **Query Processing** → Optional HyDE (Hypothetical Document Embeddings) or Decomposition
3. **Retrieval** → FAISS index search using embeddings
4. **Context Assembly** → Top-k documents assembled with metadata
5. **Generation** → LLM generates answer with citations
6. **Evaluation** → Optional metrics calculation (relevance, citations, groundedness)

### Retrieval Strategies

The benchmark scripts can compare multiple retrieval strategies:
- **Baseline**: Simple vector search
- **HyDE**: Generate hypothetical answer document and search by that
- **Decomposition**: Break complex queries into sub-queries
- **Hybrid**: Combine multiple strategies

### Performance Optimization

For production deployment, consider:
- Using HNSW index type for faster search (vs FlatL2)
- Implementing query caching
- Using smaller embedding models for speed
- Adding query pre-filtering by code/article type
