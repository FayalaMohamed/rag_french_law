# AGENTS.md - French Legal RAG System

This document provides guidelines for agentic coding agents working on this French Legal RAG (Retrieval-Augmented Generation) codebase.

## Project Overview

A RAG system for French legal document question answering using:
- **Language**: Python 3.10-3.12
- **Vector Store**: FAISS for document retrieval
- **Embeddings**: sentence-camembert-base (French BERT)
- **LLM**: Ollama (ministral-3:3b-instruct)
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
pytest tests/test_citation_manager.py

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

### Environment Setup
```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
- `OPENAI_API_KEY`: OpenAI API key
- `HF_TOKEN`: HuggingFace token (optional)
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)

## Code Style Guidelines

### Import Organization
Order imports in this sequence with blank lines between groups:
1. Standard library imports (`os`, `sys`, `json`, `re`, `typing`)
2. Third-party imports (`langchain`, `faiss`, `pytest`, `datasets`)
3. Local application imports (relative imports with `.`)

```python
import os
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
- **Classes**: PascalCase (`RAGPipeline`, `CitationManager`)
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
├── chains/              # RAG chain implementations
│   ├── llm_chain.py    # LLM wrapper
│   └── rag_chain.py    # Main pipeline
├── data/                # Data loading and storage
│   ├── loader.py       # Dataset loading
│   └── vector_store.py # FAISS vector store
├── models/              # Model wrappers
│   └── embeddings.py   # Embedding model
├── prompts/             # Prompt templates
│   └── prompts.py
├── utils/               # Utility functions
│   ├── text_processing.py
│   ├── citation_manager.py
│   ├── query_classifier.py
│   ├── environment.py
│   └── evaluation.py
└── tests/               # Test files (mirrors utils structure)
```

### Testing Conventions
- Use `pytest` framework
- Test class naming: `Test{ClassName}`
- Setup method: `setup_method(self)` for fixtures
- Test naming: `test_{method_name}_{scenario}`
- Import path: `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`

```python
class TestCitationManager:
    def setup_method(self):
        self.manager = CitationManager()

    def test_parse_article_citation(self):
        citation = self.manager.parse_citation("Article L1234-1 du Code du travail")
        assert citation.article == "L1234-1"
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

### French Text Handling
- Always use UTF-8 encoding (`encoding='utf-8'`)
- Normalize special characters (œ, æ, «», …)
- Use French punctuation patterns for sentence splitting
- Handle accented characters in regex patterns
