"""
Benchmark script for text processing configurations.
Tests different chunk sizes and overlap strategies by building real indexes and measuring retrieval performance.
"""

import os
import sys
import json
import time
import gc
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_processing import LegalTextProcessor, chunk_documents
from utils.evaluation import RAGEvaluator
from data.loader import load_french_legal_data, filter_active_articles, preprocess_articles
from data.vector_store import FAISSVectorStore
from models.embeddings import EmbeddingModel
from dotenv import load_dotenv
import config


try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    device = "cpu"


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    name: str
    chunk_size: int
    chunk_overlap: int
    strategy: str  # "semantic" or "overlapping"


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: ChunkingConfig
    num_chunks: int
    avg_chunk_size: float
    build_time_seconds: float
    index_size_mb: float
    retrieval_metrics: Dict
    latency_ms: float


# Strategic configurations to test
BENCHMARK_CONFIGS = [
    ChunkingConfig("semantic_small", 256, 0, "semantic"),
    ChunkingConfig("semantic_medium", 500, 50, "semantic"),
    ChunkingConfig("semantic_large", 1000, 100, "semantic"),
    ChunkingConfig("semantic_xlarge", 2000, 200, "semantic"),
    ChunkingConfig("overlap_small", 500, 100, "overlapping"),
    ChunkingConfig("overlap_medium", 1000, 200, "overlapping"),
    ChunkingConfig("overlap_large", 1000, 500, "overlapping"),
]





class TextProcessingBenchmark:
    """Benchmark for text processing configurations using real indexes."""
    
    def __init__(self, dataset_limit: int = 1000, output_dir: str = "data/benchmark_indexes"):
        """
        Initialize benchmark.
        
        Args:
            dataset_limit: Max number of articles to load (for faster testing)
            output_dir: Directory to save benchmark indexes (default: data/benchmark_indexes)
        """
        self.evaluator = RAGEvaluator()
        self.dataset_limit = dataset_limit
        self.embedding_model = None
        self.documents = None
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Indexes will be saved to: {os.path.abspath(self.output_dir)}")
        
    def load_data(self):
        """Load the French legal dataset."""
        print("\n" + "="*80)
        print("LOADING DATASET")
        print("="*80)
        
        load_dotenv()
        
        print("\n1. Loading dataset...")
        dataset = load_french_legal_data(config.DATASET_NAME)
        
        print("\n2. Filtering active articles...")
        filtered = filter_active_articles(dataset)
        
        print("\n3. Preprocessing articles...")
        articles = preprocess_articles(filtered)
        
        if self.dataset_limit and len(articles) > self.dataset_limit:
            print(f"\n4. Limiting to {self.dataset_limit} articles for benchmarking")
            articles = articles[:self.dataset_limit]
        
        self.documents = articles
        print(f"\n✓ Loaded {len(articles)} articles for testing")
        
        # Generate test queries from actual documents
        self._generate_test_queries()
        
    def _generate_test_queries(self):
        """Generate test queries based on actual documents in the dataset."""
        import random
        random.seed(42)  # For reproducibility
        
        if self.documents is None:
            raise ValueError("Documents not loaded")
        
        # Select 5 random documents for testing
        test_docs = random.sample(self.documents, min(5, len(self.documents)))
        
        self.test_queries = []
        for i, doc in enumerate(test_docs):
            article = doc.get("article", "")
            code = doc.get("code", "")
            text = doc.get("text", "")
            
            # Extract first sentence or first 100 chars for context
            first_sentence = text.split('.')[0] if text else ""
            if len(first_sentence) > 150:
                first_sentence = first_sentence[:150] + "..."
            
            # Create query based on document content
            query = f"Article {article} du {code}: {first_sentence}"
            
            self.test_queries.append({
                "query": query,
                "relevant_articles": [article],
                "expected_keywords": article.split('-') if article else [],
                "doc_id": doc.get("id", ""),
                "in_context": True
            })
        
        print(f"✓ Generated {len(self.test_queries)} test queries from actual documents")
        
    def init_embedding_model(self):
        """Initialize the embedding model once."""
        print(f"\n5. Initializing embedding model (device: {device})...")
        self.embedding_model = EmbeddingModel(device=device)
        print("✓ Embedding model ready")
        
    def chunk_documents_with_config(
        self, 
        documents: Optional[List[Dict]], 
        config: ChunkingConfig
    ) -> List[Dict]:
        if documents is None:
            raise ValueError("Documents not loaded")
        """
        Chunk documents using the specified configuration.
        
        Args:
            documents: List of document dicts
            config: Chunking configuration
            
        Returns:
            List of chunked documents
        """
        processor = LegalTextProcessor(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        chunked_docs = []
        
        for doc in documents:
            text = doc.get("text", "")
            if not text:
                continue
                
            metadata = {
                "article": doc.get("article", ""),
                "code": doc.get("code", ""),
                "id": doc.get("id", "")
            }
            
            if config.strategy == "semantic":
                chunks = processor.create_semantic_chunks(text, metadata=metadata)
                for chunk in chunks:
                    chunked_docs.append({
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "chunk_index": chunk.chunk_index
                    })
            else:  # overlapping
                # For overlapping, chunk per document then combine
                doc_chunks = processor.create_overlapping_chunks(text, metadata=metadata)
                for chunk in doc_chunks:
                    chunked_docs.append({
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "chunk_index": chunk.chunk_index
                    })
        
        return chunked_docs
    
    def build_index_for_config(
        self, 
        config: ChunkingConfig
    ) -> Tuple[FAISSVectorStore, float, int, float, float]:
        """
        Build a FAISS index for a specific chunking configuration.
        
        Args:
            config: Chunking configuration
            
        Returns:
            Tuple of (vector_store, build_time_seconds, num_chunks, avg_size, index_size_mb)
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model not initialized")
        print(f"\n  Chunking documents with config: {config.name}")
        start_time = time.time()
        
        # Chunk the documents
        chunked_docs = self.chunk_documents_with_config(self.documents, config)
        chunking_time = time.time() - start_time
        
        num_chunks = len(chunked_docs)
        print(f"    Generated {num_chunks} chunks")
        
        # Calculate average chunk size
        avg_size = sum(len(d["text"]) for d in chunked_docs) / max(num_chunks, 1)
        print(f"    Average chunk size: {avg_size:.0f} chars")
        
        # Build index
        print(f"  Building FAISS index...")
        index_start = time.time()
        
        index_path = os.path.join(
            self.output_dir, 
            f"index_{config.name}_{config.chunk_size}_{config.chunk_overlap}.faiss"
        )
        
        if self.embedding_model is None or self.embedding_model.embedding_dim is None:
            raise ValueError("Embedding model not properly initialized")
        
        vector_store = FAISSVectorStore(
            embedding_dim=self.embedding_model.embedding_dim,
            index_path=index_path
        )
        
        # Extract texts and metadata
        texts = [doc["text"] for doc in chunked_docs]
        metadatas = [doc["metadata"] for doc in chunked_docs]
        
        # Embed and add to index
        print(f"  Embedding {len(texts)} chunks...")
        embeddings = self.embedding_model.embed_documents(texts)
        vector_store.add_documents(texts, embeddings, metadatas)
        
        # Save index
        vector_store.save_index(index_path)
        
        build_time = time.time() - start_time
        
        # Get index size
        index_size = os.path.getsize(index_path) / (1024 * 1024)  # MB
        
        print(f"  ✓ Index built in {build_time:.2f}s")
        print(f"  ✓ Index size: {index_size:.2f} MB")
        
        return vector_store, build_time, num_chunks, avg_size, index_size
    
    def run_retrieval_tests(
        self, 
        vector_store: FAISSVectorStore, 
        k: int = 5
    ) -> Tuple[List[Dict], float]:
        """
        Run retrieval tests against the index.
        
        Args:
            vector_store: FAISS index to test
            k: Number of results to retrieve
            
        Returns:
            Tuple of (list of results per query, avg_latency_ms)
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model not initialized")
        
        results = []
        total_latency = 0
        
        test_queries = self.test_queries if hasattr(self, 'test_queries') else []
        
        for test_case in test_queries:
            query = test_case["query"]
            
            # Embed query
            start_time = time.time()
            query_embedding = self.embedding_model.embed_query(query)
            
            # Search - returns (documents, distances, metadatas)
            docs, distances, metadatas = vector_store.search(query_embedding, k=k)
            latency = (time.time() - start_time) * 1000  # ms
            total_latency += latency
            
            # Get retrieved documents
            retrieved_docs = []
            for doc_text, distance, metadata in zip(docs, distances, metadatas):
                retrieved_docs.append({
                    "text": doc_text,
                    "metadata": metadata,
                    "distance": float(distance)
                })
            
            results.append({
                "query": query,
                "retrieved_docs": retrieved_docs,
                "relevant_articles": test_case.get("relevant_articles", []),
                "latency_ms": latency
            })
        
        avg_latency = total_latency / len(test_queries) if test_queries else 0
        return results, avg_latency
    
    def evaluate_retrieval(
        self, 
        retrieval_results: List[Dict]
    ) -> Dict:
        """
        Evaluate retrieval quality.
        
        Args:
            retrieval_results: Results from retrieval tests
            
        Returns:
            Aggregated metrics
        """
        all_metrics = []
        
        for result in retrieval_results:
            relevant = result["relevant_articles"]
            retrieved = [
                doc["metadata"].get("article", "") 
                for doc in result["retrieved_docs"]
            ]
            
            # Compute metrics
            metrics = self.evaluator.retrieval_evaluator.evaluate_retrieval(
                relevant_docs=relevant,
                retrieved_docs=retrieved
            )
            all_metrics.append(asdict(metrics))
        
        # Aggregate
        aggregated = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    aggregated[key] = sum(values) / len(values)
        
        return aggregated
    
    def run_config(self, config: ChunkingConfig) -> BenchmarkResult:
        """
        Run full benchmark for a configuration.
        
        Args:
            config: Chunking configuration
            
        Returns:
            BenchmarkResult
        """
        print(f"\n{'='*80}")
        print(f"BENCHMARKING: {config.name}")
        print(f"  Strategy: {config.strategy}")
        print(f"  Chunk size: {config.chunk_size}, Overlap: {config.chunk_overlap}")
        print('='*80)
        
        # Build index
        vector_store, build_time, num_chunks, avg_size, index_size = \
            self.build_index_for_config(config)
        
        # Run retrieval tests
        num_queries = len(self.test_queries) if hasattr(self, 'test_queries') else 0
        print(f"\n  Running {num_queries} retrieval queries...")
        retrieval_results, avg_latency = self.run_retrieval_tests(vector_store)
        
        # Evaluate
        metrics = self.evaluate_retrieval(retrieval_results)
        
        print(f"\n  Results:")
        print(f"    Precision@K: {metrics.get('precision_at_k', 0):.3f}")
        print(f"    Recall@K: {metrics.get('recall_at_k', 0):.3f}")
        print(f"    MRR@K: {metrics.get('mrr_at_k', 0):.3f}")
        print(f"    Avg latency: {avg_latency:.2f} ms")
        
        # Clean up memory
        self._cleanup_memory(vector_store)
        
        return BenchmarkResult(
            config=config,
            num_chunks=num_chunks,
            avg_chunk_size=avg_size,
            build_time_seconds=build_time,
            index_size_mb=index_size,
            retrieval_metrics=metrics,
            latency_ms=avg_latency
        )
    
    def _cleanup_memory(self, vector_store):
        """Free memory after each configuration."""
        print("  Cleaning up memory...")
        
        # Delete the vector store and its index
        if vector_store:
            # Clear FAISS index
            if hasattr(vector_store, 'index') and vector_store.index:
                del vector_store.index
                vector_store.index = None
            # Clear stored data
            if hasattr(vector_store, 'documents'):
                vector_store.documents = []
            if hasattr(vector_store, 'metadatas'):
                vector_store.metadatas = []
            del vector_store
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU memory if using CUDA
        if device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"  ✓ GPU memory cleared: {torch.cuda.memory_allocated() / 1024**2:.1f} MB allocated")
            except:
                pass
        
        print("  ✓ Memory cleanup complete")
    
    def run_all(self, config_names: Optional[List[str]] = None) -> List[BenchmarkResult]:
        """
        Run benchmark for all or selected configurations.
        
        Args:
            config_names: Optional list of config names to test (default: all)
            
        Returns:
            List of BenchmarkResult
        """
        # Load data and init model once
        if not self.documents:
            self.load_data()
        if not self.embedding_model:
            self.init_embedding_model()
        
        # Select configs
        if config_names and 'all' not in config_names:
            configs = [c for c in BENCHMARK_CONFIGS if c.name in config_names]
        else:
            configs = BENCHMARK_CONFIGS
        
        results = []
        for config in configs:
            try:
                result = self.run_config(config)
                results.append(result)
            except Exception as e:
                print(f"\n  ✗ Error testing {config.name}: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    
    def generate_report(self, results: List[BenchmarkResult], output_path: Optional[str] = None):
        """Generate benchmark report."""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        # Summary table
        print("\n{:<18} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "Config", "Chunks", "AvgSize", "Build(s)", "Index(MB)", "Prec@K", "Recall@K", "Latency(ms)"
        ))
        print("-"*100)
        
        for result in results:
            m = result.retrieval_metrics
            print("{:<18} {:>8} {:>10.0f} {:>10.2f} {:>10.2f} {:>10.3f} {:>10.3f} {:>10.2f}".format(
                result.config.name,
                result.num_chunks,
                result.avg_chunk_size,
                result.build_time_seconds,
                result.index_size_mb,
                m.get('precision_at_k', 0),
                m.get('recall_at_k', 0),
                result.latency_ms
            ))
        
        # Detailed breakdown
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        
        for result in results:
            m = result.retrieval_metrics
            print(f"\n{result.config.name} ({result.config.strategy}):")
            print(f"  Chunking: {result.config.chunk_size} chars, {result.config.chunk_overlap} overlap")
            print(f"  Chunks: {result.num_chunks} (avg {result.avg_chunk_size:.0f} chars)")
            print(f"  Build time: {result.build_time_seconds:.2f}s")
            print(f"  Index size: {result.index_size_mb:.2f} MB")
            print(f"  Retrieval:")
            print(f"    Precision@K: {m.get('precision_at_k', 0):.3f}")
            print(f"    Recall@K: {m.get('recall_at_k', 0):.3f}")
            print(f"    MRR@K: {m.get('mrr_at_k', 0):.3f}")
            print(f"    Hit Rate@K: {m.get('hit_rate_at_k', 0):.3f}")
            print(f"  Latency: {result.latency_ms:.2f} ms/query")
        
        # Best configs
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        if results:
            # Best retrieval
            best_retrieval = max(results, 
                               key=lambda r: r.retrieval_metrics.get('precision_at_k', 0) + 
                                            r.retrieval_metrics.get('recall_at_k', 0))
            print(f"\n🏆 Best Retrieval Quality: {best_retrieval.config.name}")
            print(f"   Precision: {best_retrieval.retrieval_metrics.get('precision_at_k', 0):.3f}, "
                  f"Recall: {best_retrieval.retrieval_metrics.get('recall_at_k', 0):.3f}")
            
            # Fastest
            fastest = min(results, key=lambda r: r.latency_ms)
            print(f"\n⚡ Fastest Retrieval: {fastest.config.name}")
            print(f"   {fastest.latency_ms:.2f} ms/query")
            
            # Most efficient (fewest chunks with good precision)
            good_precision = [r for r in results 
                            if r.retrieval_metrics.get('precision_at_k', 0) >= 0.3]
            if good_precision:
                efficient = min(good_precision, key=lambda r: r.num_chunks)
                print(f"\n💾 Most Efficient: {efficient.config.name}")
                print(f"   {efficient.num_chunks} chunks, good precision")
            
            # Smallest index
            smallest = min(results, key=lambda r: r.index_size_mb)
            print(f"\n📦 Smallest Index: {smallest.config.name}")
            print(f"   {smallest.index_size_mb:.2f} MB")
        
        # Save to file
        if output_path:
            num_queries = len(self.test_queries) if hasattr(self, 'test_queries') else 0
            report = {
                "timestamp": datetime.now().isoformat(),
                "dataset_limit": self.dataset_limit,
                "num_test_queries": num_queries,
                "results": [
                    {
                        "config": {
                            "name": r.config.name,
                            "chunk_size": r.config.chunk_size,
                            "chunk_overlap": r.config.chunk_overlap,
                            "strategy": r.config.strategy
                        },
                        "num_chunks": r.num_chunks,
                        "avg_chunk_size": r.avg_chunk_size,
                        "build_time_seconds": r.build_time_seconds,
                        "index_size_mb": r.index_size_mb,
                        "retrieval_metrics": r.retrieval_metrics,
                        "latency_ms": r.latency_ms
                    }
                    for r in results
                ]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"\n\n💾 Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark text processing configurations by building real indexes"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of articles to use for benchmarking (default: 500)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_text_processing_results.json",
        help="Path to save results (default: benchmark_text_processing_results.json)"
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs='+',
        choices=[c.name for c in BENCHMARK_CONFIGS] + ['all'],
        default=['all'],
        help="Configurations to test"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with 2 configs only"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmark_indexes",
        help="Directory to save benchmark indexes (default: data/benchmark_indexes)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("TEXT PROCESSING BENCHMARK - REAL INDEX BUILDING")
    print("="*80)
    print(f"\nDataset limit: {args.limit} articles")
    print(f"Device: {device}")
    
    # Quick mode
    if args.quick:
        print("\n⚡ QUICK MODE: Testing only semantic_medium and overlap_medium")
        args.configs = ["semantic_medium", "overlap_medium"]
    
    # Run benchmark
    benchmark = TextProcessingBenchmark(dataset_limit=args.limit, output_dir=args.output_dir)
    results = benchmark.run_all(args.configs)
    
    # Generate report
    benchmark.generate_report(results, args.output)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
