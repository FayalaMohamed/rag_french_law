"""
FAISS Index Performance Benchmarking Script

Compares different FAISS index types for the French Legal RAG system.
Generates performance metrics and comparison graphs.

Usage:
    python benchmark_faiss_indices.py --dataset_size 1000 --queries 100
    python benchmark_faiss_indices.py --use_existing_index data/faiss_index --queries 100
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import argparse
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.vector_store import FAISSVectorStore
from models.embeddings import EmbeddingModel
from data.loader import load_french_legal_data, preprocess_articles, filter_active_articles
import config


def load_legal_documents(max_documents: int = None):
    dataset = load_french_legal_data(config.DATASET_NAME)
    filtered = filter_active_articles(dataset)
    articles = preprocess_articles(filtered)
    
    if max_documents:
        articles = articles[:max_documents]
    
    return [article["text"] for article in articles]


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single index type."""
    index_type: str
    build_time: float
    avg_search_time: float
    total_search_time: float
    memory_usage_mb: float
    index_size_mb: float
    recall_at_k: float
    total_documents: int
    nqueries: int
    k: int
    
    # Index-specific parameters
    nlist: int = None
    nprobe: int = None
    nbits: int = None
    ef_construction: int = None
    ef_search: int = None
    m: int = None
    pq_m: int = None
    pq_nbits: int = None


class FAISSBenchmark:    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        documents: List[str],
        queries: List[str],
        k: int = 5
    ):
        self.embedding_model = embedding_model
        self.documents = documents
        self.queries = queries
        self.k = k
        self.embedding_dim = embedding_model.embedding_dim
        
        print(f"Computing embeddings for {len(documents)} documents...")
        self.doc_embeddings = embedding_model.embed_documents(documents)
        
        print(f"Computing embeddings for {len(queries)} queries...")
        self.query_embeddings = embedding_model.embed_documents(queries)
    
    def benchmark_index(
        self,
        index_type: str,
        **index_params
    ) -> BenchmarkResult:
        """Benchmark a single index type."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {index_type}")
        print(f"{'='*60}")
        
        # Create index
        vector_store = FAISSVectorStore(
            embedding_dim=self.embedding_dim,
            index_type=index_type,
            **index_params
        )
        
        # Measure build time
        print(f"Building index with {len(self.documents)} documents...")
        start_time = time.time()
        vector_store.add_documents(self.documents, self.doc_embeddings)
        build_time = time.time() - start_time
        print(f"Build time: {build_time:.2f}s")
        
        # Measure search time
        print(f"Running {len(self.queries)} queries...")
        start_time = time.time()
        all_results = []
        for query_emb in self.query_embeddings:
            docs, dists, metas = vector_store.search(query_emb, k=self.k)
            all_results.append(docs)
        total_search_time = time.time() - start_time
        avg_search_time = total_search_time / len(self.queries)
        print(f"Total search time: {total_search_time:.4f}s")
        print(f"Avg search time per query: {avg_search_time:.4f}s")
        
        # Measure memory usage (approximate)
        index_size_bytes = vector_store.index.ntotal * self.embedding_dim * 4  # float32
        memory_usage_mb = index_size_bytes / (1024 * 1024)
        
        # Save and measure index size on disk
        import tempfile
        temp_dir = tempfile.mkdtemp()
        vector_store.save_index(temp_dir)
        
        index_size_mb = 0
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            index_size_mb += os.path.getsize(file_path) / (1024 * 1024)
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"Memory usage: {memory_usage_mb:.2f} MB")
        print(f"Index size on disk: {index_size_mb:.2f} MB")
        
        # Get stats
        stats = vector_store.get_stats()
        
        return BenchmarkResult(
            index_type=index_type,
            build_time=build_time,
            avg_search_time=avg_search_time,
            total_search_time=total_search_time,
            memory_usage_mb=memory_usage_mb,
            index_size_mb=index_size_mb,
            recall_at_k=0.0,  # Will be calculated later
            total_documents=len(self.documents),
            nqueries=len(self.queries),
            k=self.k,
            nlist=stats.get("nlist"),
            nprobe=stats.get("nprobe"),
            nbits=stats.get("nbits"),
            ef_construction=stats.get("ef_construction"),
            ef_search=stats.get("ef_search"),
            m=stats.get("m"),
            pq_m=stats.get("pq_m"),
            pq_nbits=stats.get("pq_nbits")
        )
    
    def calculate_recall(
        self,
        index_type: str,
        ground_truth_results: List[List[str]],
        **index_params
    ) -> float:
        """Calculate recall@k compared to flat_l2 baseline."""
        print(f"\nCalculating recall for {index_type}...")
        
        # Create index
        vector_store = FAISSVectorStore(
            embedding_dim=self.embedding_dim,
            index_type=index_type,
            **index_params
        )
        vector_store.add_documents(self.documents, self.doc_embeddings)
        
        # Run queries
        total_recall = 0.0
        for i, query_emb in enumerate(tqdm(self.query_embeddings, desc=f"Recall calc ({index_type})")):
            docs, _, _ = vector_store.search(query_emb, k=self.k)
            
            # Calculate intersection with ground truth
            ground_truth_set = set(ground_truth_results[i])
            retrieved_set = set(docs)
            intersection = len(ground_truth_set & retrieved_set)
            
            recall = intersection / len(ground_truth_set) if ground_truth_set else 0.0
            total_recall += recall
        
        avg_recall = total_recall / len(self.queries)
        print(f"Recall@{self.k}: {avg_recall:.4f}")
        return avg_recall

    def benchmark_variable_k(
        self,
        index_type: str,
        k_values: List[int] = None,
        **index_params
    ) -> Dict[int, Dict]:
        """Benchmark performance with different k values.

        Args:
            index_type: Type of index to benchmark
            k_values: List of k values to test (default: [1, 3, 5, 10, 20, 50])
            **index_params: Index-specific parameters

        Returns:
            Dict mapping k values to performance metrics
        """
        if k_values is None:
            k_values = [1, 3, 5, 10, 20, 50]

        print(f"\n{'='*60}")
        print(f"Variable K Benchmark: {index_type}")
        print(f"Testing k values: {k_values}")
        print(f"{'='*60}")

        # Create index once
        vector_store = FAISSVectorStore(
            embedding_dim=self.embedding_dim,
            index_type=index_type,
            **index_params
        )

        print(f"Building {index_type} index with {len(self.documents)} documents...")
        vector_store.add_documents(self.documents, self.doc_embeddings)

        # Create ground truth store (flat_l2) for recall calculation
        print("Creating ground truth store for recall calculation...")
        ground_truth_store = FAISSVectorStore(
            embedding_dim=self.embedding_dim,
            index_type="flat_l2"
        )
        ground_truth_store.add_documents(self.documents, self.doc_embeddings)

        results = {}

        for k in k_values:
            print(f"\nTesting k={k}...")

            # Benchmark search time
            start_time = time.time()
            all_results = []
            for query_emb in self.query_embeddings:
                docs, dists, metas = vector_store.search(query_emb, k=k)
                all_results.append(docs)
            search_time = time.time() - start_time
            avg_search_time = search_time / len(self.queries)

            # Calculate recall
            total_recall = 0.0
            for i, query_emb in enumerate(self.query_embeddings):
                # Get ground truth at this k
                gt_docs, _, _ = ground_truth_store.search(query_emb, k=k)

                # Calculate recall
                ground_truth_set = set(gt_docs)
                retrieved_set = set(all_results[i])
                intersection = len(ground_truth_set & retrieved_set)

                recall = intersection / len(ground_truth_set) if ground_truth_set else 0.0
                total_recall += recall

            avg_recall = total_recall / len(self.queries)

            results[k] = {
                "avg_search_time": avg_search_time,
                "total_search_time": search_time,
                "recall": avg_recall,
                "queries_per_second": len(self.queries) / search_time if search_time > 0 else 0
            }

            print(f"  Recall@{k}: {avg_recall:.4f}")
            print(f"  Avg search time: {avg_search_time:.4f}s")
            print(f"  Queries/sec: {results[k]['queries_per_second']:.1f}")

        return results

    def run_full_benchmark(self) -> List[BenchmarkResult]:
        """Run benchmarks for all supported index types."""
        results = []
        
        # First, establish ground truth with flat_l2
        print("\n" + "="*60)
        print("ESTABLISHING GROUND TRUTH (FlatL2)")
        print("="*60)
        
        flat_result = self.benchmark_index("flat_l2")
        
        # Get ground truth results from flat_l2
        ground_truth_store = FAISSVectorStore(
            embedding_dim=self.embedding_dim,
            index_type="flat_l2"
        )
        ground_truth_store.add_documents(self.documents, self.doc_embeddings)
        
        ground_truth_results = []
        for query_emb in self.query_embeddings:
            docs, _, _ = ground_truth_store.search(query_emb, k=self.k)
            ground_truth_results.append(docs)
        
        flat_result.recall_at_k = 1.0  # Perfect recall by definition
        results.append(flat_result)
        
        # Define index configurations to test
        index_configs = [
            # IVF configurations
            ("ivf_flat", {"nlist": 100, "nprobe": 10}),
            ("ivf_flat", {"nlist": 200, "nprobe": 20}),
            ("ivf_pq", {"nlist": 100, "nprobe": 10, "pq_m": 8, "pq_nbits": 8}),
            
            # HNSW configurations
            ("hnsw_flat", {"m": 16, "ef_construction": 40, "ef_search": 16}),
            ("hnsw_flat", {"m": 32, "ef_construction": 80, "ef_search": 32}),
            
            # LSH
            ("lsh", {"nbits": 8}),
            ("lsh", {"nbits": 16}),
        ]
        
        # Test other indices
        for index_type, params in index_configs:
            try:
                result = self.benchmark_index(index_type, **params)
                result.recall_at_k = self.calculate_recall(
                    index_type, ground_truth_results, **params
                )
                results.append(result)
            except Exception as e:
                print(f"Error benchmarking {index_type}: {e}")
                import traceback
                traceback.print_exc()
        
        return results


def plot_results(results: List[BenchmarkResult], output_dir: str = "benchmark_results"):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    def get_label(r):
        if r.index_type == "flat_l2":
            return r.index_type
        elif r.index_type in ["ivf_flat", "ivf_pq"]:
            return f"{r.index_type}\nnlist={r.nlist}, nprobe={r.nprobe}"
        elif r.index_type == "hnsw_flat":
            return f"{r.index_type}\nm={r.m}, ef={r.ef_search}"
        elif r.index_type == "lsh":
            return f"{r.index_type}\nnbits={r.nbits}"
        return r.index_type
    
    labels = [get_label(r) for r in results]
    
    # Color mapping
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    
    # Plot 1: Build Time vs Recall
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(
        [r.build_time for r in results],
        [r.recall_at_k for r in results],
        c=range(len(results)),
        cmap='Set3',
        s=200,
        alpha=0.7,
        edgecolors='black'
    )
    
    for i, r in enumerate(results):
        ax.annotate(
            f"{r.index_type}",
            (r.build_time, r.recall_at_k),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    ax.set_xlabel('Build Time (seconds)', fontsize=12)
    ax.set_ylabel(f'Recall@{results[0].k}', fontsize=12)
    ax.set_title('Build Time vs Recall Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'build_time_vs_recall.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Search Time vs Recall
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(
        [r.avg_search_time * 1000 for r in results],  # Convert to ms
        [r.recall_at_k for r in results],
        c=range(len(results)),
        cmap='Set3',
        s=200,
        alpha=0.7,
        edgecolors='black'
    )
    
    for i, r in enumerate(results):
        ax.annotate(
            f"{r.index_type}",
            (r.avg_search_time * 1000, r.recall_at_k),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    ax.set_xlabel('Average Search Time (ms)', fontsize=12)
    ax.set_ylabel(f'Recall@{results[0].k}', fontsize=12)
    ax.set_title('Search Speed vs Recall Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'search_time_vs_recall.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Memory Usage Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(results))
    bars = ax.bar(x_pos, [r.memory_usage_mb for r in results], color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Index Type', fontsize=12)
    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_usage.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Index Size on Disk
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x_pos, [r.index_size_mb for r in results], color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Index Type', fontsize=12)
    ax.set_ylabel('Index Size (MB)', fontsize=12)
    ax.set_title('Index Size on Disk', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'index_size.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Recall Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x_pos, [r.recall_at_k for r in results], color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Index Type', fontsize=12)
    ax.set_ylabel(f'Recall@{results[0].k}', fontsize=12)
    ax.set_title(f'Recall@{results[0].k} Comparison (vs FlatL2)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Recall')
    ax.legend()
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recall_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to: {output_dir}/")


def plot_variable_k_results(
    results_dict: Dict[str, Dict[int, Dict]],
    output_dir: str = "benchmark_results"
):
    """Generate plots for variable k benchmark results.

    Args:
        results_dict: Dict mapping index type to results per k value
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all k values
    first_result = list(results_dict.values())[0]
    k_values = sorted(first_result.keys())

    # Plot 1: Recall vs K
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_dict)))

    for i, (index_type, results) in enumerate(results_dict.items()):
        recalls = [results[k]["recall"] for k in k_values]
        ax.plot(k_values, recalls, 'o-', linewidth=2, label=index_type, color=colors[i], markersize=8)

    ax.set_xlabel('K (Number of Results)', fontsize=12)
    ax.set_ylabel('Recall@K', fontsize=12)
    ax.set_title('Recall vs K for Different Index Types', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variable_k_recall.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Search Time vs K
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (index_type, results) in enumerate(results_dict.items()):
        times = [results[k]["avg_search_time"] * 1000 for k in k_values]  # Convert to ms
        ax.plot(k_values, times, 'o-', linewidth=2, label=index_type, color=colors[i], markersize=8)

    ax.set_xlabel('K (Number of Results)', fontsize=12)
    ax.set_ylabel('Avg Search Time (ms)', fontsize=12)
    ax.set_title('Search Time vs K for Different Index Types', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variable_k_search_time.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Queries per Second vs K
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (index_type, results) in enumerate(results_dict.items()):
        qps = [results[k]["queries_per_second"] for k in k_values]
        ax.plot(k_values, qps, 'o-', linewidth=2, label=index_type, color=colors[i], markersize=8)

    ax.set_xlabel('K (Number of Results)', fontsize=12)
    ax.set_ylabel('Queries per Second', fontsize=12)
    ax.set_title('Throughput vs K for Different Index Types', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variable_k_qps.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVariable k plots saved to: {output_dir}/")


def save_results_json(results: List[BenchmarkResult], output_dir: str = "benchmark_results"):
    """Save benchmark results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    results_dict = {
        "benchmark_info": {
            "total_documents": results[0].total_documents,
            "num_queries": results[0].nqueries,
            "k": results[0].k,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "results": [asdict(r) for r in results]
    }
    
    output_path = os.path.join(output_dir, "benchmark_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {output_path}")


def print_summary_table(results: List[BenchmarkResult]):
    """Print a formatted summary table."""
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY")
    print("="*100)
    
    header = f"{'Index Type':<20} {'Build(s)':<10} {'Search(ms)':<12} {'Recall@K':<10} {'Memory(MB)':<12} {'Size(MB)':<10}"
    print(header)
    print("-"*100)
    
    for r in results:
        row = (f"{r.index_type:<20} "
               f"{r.build_time:<10.2f} "
               f"{r.avg_search_time*1000:<12.2f} "
               f"{r.recall_at_k:<10.3f} "
               f"{r.memory_usage_mb:<12.1f} "
               f"{r.index_size_mb:<10.1f}")
        print(row)
    
    print("="*100)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark different FAISS index types for French Legal RAG"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=1000,
        help="Number of documents to use for benchmarking (default: 1000)"
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=100,
        help="Number of test queries to run (default: 100)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results and plots (default: benchmark_results)"
    )
    parser.add_argument(
        "--use_existing_index",
        type=str,
        default=None,
        help="Path to existing FAISS index to use for documents"
    )

    args = parser.parse_args()

    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = EmbeddingModel()

    # Load or generate documents
    if args.use_existing_index and os.path.exists(args.use_existing_index):
        print(f"Loading existing index from {args.use_existing_index}...")
        vector_store = FAISSVectorStore(
            embedding_dim=embedding_model.embedding_dim,
            index_path=args.use_existing_index
        )
        documents = vector_store.documents
        print(f"Loaded {len(documents)} documents from existing index")
    else:
        print(f"Loading {args.dataset_size} documents from dataset...")
        documents = load_legal_documents(max_documents=args.dataset_size)

    # Generate synthetic queries if needed
    print(f"Generating {args.queries} test queries...")
    # Use document snippets as queries for simplicity
    np.random.seed(42)
    query_indices = np.random.choice(len(documents), min(args.queries, len(documents)), replace=False)
    queries = [documents[i][:200] for i in query_indices]  # First 200 chars as query

    # Run benchmarks
    benchmark = FAISSBenchmark(
        embedding_model=embedding_model,
        documents=documents,
        queries=queries,
        k=args.k
    )

    # Run standard benchmark
    results = benchmark.run_full_benchmark()

    # Print summary
    print_summary_table(results)

    # Save results
    save_results_json(results, args.output_dir)

    # Generate plots
    plot_results(results, args.output_dir)

    # Run variable k benchmark (always included)
    print("\n" + "="*60)
    print("RUNNING VARIABLE K BENCHMARK")
    print("="*60)

    # Test a subset of index types with variable k
    index_configs = [
        ("flat_l2", {}),
        ("ivf_flat", {"nlist": 100, "nprobe": 10}),
        ("hnsw_flat", {"m": 16, "ef_construction": 40, "ef_search": 16}),
    ]

    variable_k_results = {}
    for index_type, params in index_configs:
        try:
            results = benchmark.benchmark_variable_k(index_type, **params)
            variable_k_results[index_type] = results
        except Exception as e:
            print(f"Error benchmarking {index_type}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary table
    print("\n" + "="*80)
    print("VARIABLE K BENCHMARK SUMMARY")
    print("="*80)
    for index_type, results in variable_k_results.items():
        print(f"\n{index_type}:")
        print(f"{'K':<6} {'Recall':<10} {'Time(ms)':<12} {'QPS':<10}")
        print("-"*40)
        for k in sorted(results.keys()):
            r = results[k]
            print(f"{k:<6} {r['recall']:<10.3f} {r['avg_search_time']*1000:<12.2f} {r['queries_per_second']:<10.1f}")

    # Generate plots
    plot_variable_k_results(variable_k_results, args.output_dir)

    # Save results to JSON
    output_path = os.path.join(args.output_dir, "variable_k_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(variable_k_results, f, ensure_ascii=False, indent=2)
    print(f"\nVariable k results saved to: {output_path}")

    print(f"\n{'='*60}")
    print("Benchmark complete!")
    print(f"Results saved to: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
