 Based on your codebase analysis, here are **additional dimensions** you can test, tweak, and compare:

## 🎯 **Retrieval Strategies**
- **HyDE (Hypothetical Document Embeddings)** - on/off toggle
- **Query Decomposition** - split complex queries into sub-questions
- **Variable top_k** - adjust per query type (3-50)
- **Reranking** - enable/disable result reranking by query type

## 🔧 **Text Processing**
- **Chunk sizes**: 256, 500, 1000, 2000 characters
- **Chunk overlap**: 0, 50, 100, 200, 500
- **Semantic vs. overlapping chunks** - two different chunking strategies

## 📊 **Query Classification**
- **9 query types** with different retrieval parameters:
  - DEFINITION (k=3, no HyDE)
  - PROCEDURAL (k=7, with decomposition)
  - COMPARATIVE (k=10, with reranking)
  - etc.

## 🎛️ **FAISS Index Fine-tuning**
Beyond index types, you can tune:
- **HNSW**: `m` (8-64), `ef_construction` (40-200), `ef_search` (16-128)
- **IVF**: `nlist` (50-500), `nprobe` (5-50)
- **IVF_PQ**: `pq_m` (4-16), `pq_nbits` (8-16)

## 🧪 **Evaluation Metrics**
- **Retrieval**: precision@k, recall@k, MRR, NDCG, hit rate
- **Generation**: relevance, citation precision/recall, groundedness, completeness

Want me to help you set up A/B tests for any of these?