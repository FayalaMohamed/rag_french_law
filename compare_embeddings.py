"""
Embedding Model Comparison Script for French Legal RAG

This script compares multiple embedding models on the same test queries
to evaluate retrieval quality, cross-model agreement, and LLM-as-judge performance.

Usage:
    python compare_embeddings.py --build-indexes --limit 5000
    python compare_embeddings.py --compare-queries --queries-file test_queries.json
    python compare_embeddings.py --full-evaluation --limit 5000

Note: Since the index contains only 5000 documents, some queries may have no relevant
matches. The evaluation metrics account for this by measuring confidence, distance 
distributions, and semantic coherence.
"""

import os
import sys

# Set PyTorch CUDA memory configuration to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime
from scipy.stats import spearmanr, kendalltau
from dotenv import load_dotenv

from data.loader import (
    load_french_legal_data,
    filter_active_articles,
    preprocess_articles,
    save_processed_data,
    load_processed_data
)
from models.embeddings import EmbeddingModel
from data.vector_store import FAISSVectorStore
from chains.rag_chain import RAGPipeline
from chains.llm_chain import LLMChainWrapper
import config


@dataclass
class ModelResult:
    """Results for a single model on a single query"""
    model_name: str
    query: str
    query_category: str
    retrieved_docs: List[str]
    distances: List[float]
    metadatas: List[Dict]
    query_time_ms: float
    answer: Optional[str] = None


@dataclass
class ComparisonMetrics:
    """Metrics comparing two models"""
    model_a: str
    model_b: str
    overlap_at_k: Dict[int, float]
    spearman_correlation: float
    kendall_tau: float
    distance_correlation: float
    semantic_coherence_a: float
    semantic_coherence_b: float


# Embedding models to test (tested on 4GB GPU)
# Note: multilingual-e5-base removed - causes OOM on 4GB GPU
TEST_MODELS = {
    # French-specific models (4GB GPU compatible)
    "sentence-camembert-base": "dangvantuan/sentence-camembert-base",
    # Multilingual models (good for French, 4GB-compatible)
    "paraphrase-multilingual-MiniLM": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "multilingual-e5-small": "intfloat/multilingual-e5-small",
}

# Test queries covering different legal domains and difficulty levels
DEFAULT_TEST_QUERIES = [
    # In-domain queries (likely in the 5000 docs)
    {
        "query": "Quelles sont les conditions de la période d'essai en droit du travail?",
        "category": "labor_law",
        "expected_domain": "Code du travail"
    },
    {
        "query": "Comment rompre un contrat de travail à durée déterminée?",
        "category": "labor_law",
        "expected_domain": "Code du travail"
    },
    {
        "query": "Quelles sont les obligations de l'employeur en matière de sécurité?",
        "category": "labor_law",
        "expected_domain": "Code du travail"
    },
    {
        "query": "Qu'est-ce que le harcèlement moral au travail?",
        "category": "labor_law",
        "expected_domain": "Code du travail"
    },
    {
        "query": "Article 1240 du Code civil définition du dommage",
        "category": "civil_law",
        "expected_domain": "Code civil"
    },
    {
        "query": "Conditions de validité d'un contrat selon le Code civil",
        "category": "civil_law",
        "expected_domain": "Code civil"
    },
    {
        "query": "Délai de prescription en droit civil",
        "category": "civil_law",
        "expected_domain": "Code civil"
    },
    {
        "query": "Responsabilité civile et faute",
        "category": "civil_law",
        "expected_domain": "Code civil"
    },
    # Out-of-domain queries (unlikely to be in the 5000 docs)
    {
        "query": "Quelles sont les règles de succession en cas de décès sans testament?",
        "category": "inheritance",
        "expected_domain": "Code civil"
    },
    {
        "query": "Procédure de divorce par consentement mutuel",
        "category": "family_law",
        "expected_domain": "Code civil"
    },
    {
        "query": "Droits du locataire en cas de vente du logement",
        "category": "housing",
        "expected_domain": "Code civil"
    },
    {
        "query": "Protection des données personnelles RGPD",
        "category": "data_protection",
        "expected_domain": "Unknown"
    },
    # Edge cases
    {
        "query": "Article L1234-1",
        "category": "citation",
        "expected_domain": "Code du travail"
    },
    {
        "query": "Code du travail période d'essai durée maximale",
        "category": "keyword_search",
        "expected_domain": "Code du travail"
    },
    {
        "query": "Comment ça marche le licenciement?",
        "category": "colloquial",
        "expected_domain": "Code du travail"
    }
]


def get_index_path(model_name: str) -> str:
    """Get the index path for a specific model"""
    # Clean model name for filesystem
    clean_name = model_name.replace("/", "_").replace("-", "_")
    return f"data/faiss_index_{clean_name}"


def build_model_index(
    model_name: str,
    model_path: str,
    articles: List[Dict],
    index_type: str = "hnsw_flat"
) -> Tuple[FAISSVectorStore, float]:
    """
    Build FAISS index for a specific embedding model.
    
    Returns:
        Tuple of (vector_store, build_time_seconds)
    """
    print(f"\n{'='*60}")
    print(f"Building index for: {model_name}")
    print(f"Model path: {model_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Initialize embedding model
    print("\n1. Loading embedding model...")
    embedding_model = EmbeddingModel(model_name=model_path)
    
    # Create vector store
    print("\n2. Creating vector store...")
    index_path = get_index_path(model_name)
    vector_store = FAISSVectorStore(
        embedding_dim=embedding_model.embedding_dim,
        index_path=index_path,
        index_type=index_type
    )
    
    # Embed documents
    print("\n3. Embedding documents...")
    texts = [article["text"] for article in articles]
    metadatas = [article["metadata"] for article in articles]
    
    embeddings = embedding_model.embed_documents(texts)
    
    # Add to index
    print("\n4. Adding to index...")
    vector_store.add_documents(texts, embeddings, metadatas)
    
    # Save index
    print("\n5. Saving index...")
    vector_store.save_index(index_path)
    
    build_time = time.time() - start_time
    print(f"\n✓ Index built successfully in {build_time:.2f} seconds")
    print(f"  Total documents: {vector_store.index.ntotal}")
    print(f"  Saved to: {index_path}")
    
    # Clean up GPU memory
    print("\n6. Cleaning up GPU memory...")
    del embedding_model
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("  ✓ GPU memory cleared")
    
    return vector_store, build_time


def build_all_indexes(
    models: Dict[str, str],
    articles: List[Dict],
    index_type: str = "hnsw_flat"
) -> Dict[str, Dict]:
    """Build indexes for all test models"""
    results = {}
    
    for model_name, model_path in models.items():
        vector_store, build_time = build_model_index(
            model_name, model_path, articles, index_type
        )
        results[model_name] = {
            "status": "success",
            "build_time": build_time,
            "index_path": get_index_path(model_name),
            "num_docs": vector_store.index.ntotal,
            "embedding_dim": vector_store.embedding_dim
        }
    
    return results


def run_queries_on_model(
    model_name: str,
    model_path: str,
    queries: List[Dict],
    k: int = 5
) -> List[ModelResult]:
    """
    Run all test queries on a specific model.
    
    Returns:
        List of ModelResult objects
    """
    print(f"\nRunning queries on {model_name}...")
    
    # Load model and index
    embedding_model = EmbeddingModel(model_name=model_path)
    
    index_path = get_index_path(model_name)
    # Check for index directory (FAISSVectorStore saves as directory with index.faiss inside)
    if not os.path.exists(index_path) or not os.path.exists(os.path.join(index_path, "index.faiss")):
        print(f"Index not found for {model_name} at {index_path}, skipping...")
        return []
    
    vector_store = FAISSVectorStore(
        embedding_dim=embedding_model.embedding_dim,
        index_path=index_path
    )
    
    results = []
    for query_data in queries:
        query = query_data["query"]
        category = query_data["category"]
        
        # Time the query
        start_time = time.time()
        docs, distances, metadatas = vector_store.search_by_text(
            query, embedding_model, k=k
        )
        query_time = (time.time() - start_time) * 1000  # Convert to ms
        
        result = ModelResult(
            model_name=model_name,
            query=query,
            query_category=category,
            retrieved_docs=docs,
            distances=distances,
            metadatas=metadatas,
            query_time_ms=query_time
        )
        results.append(result)
    
    print(f"  [OK] Completed {len(results)} queries")
    
    # Clean up GPU memory
    print(f"  Cleaning up GPU memory for {model_name}...")
    del embedding_model
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  [OK] GPU memory cleared for {model_name}")
    
    return results


def calculate_overlap_at_k(
    results_a: List[ModelResult],
    results_b: List[ModelResult],
    k_values: List[int] = [1, 3, 5]
) -> Dict[int, float]:
    """Calculate overlap between two models at different k values"""
    overlaps = {k: [] for k in k_values}
    
    for res_a, res_b in zip(results_a, results_b):
        # Create sets of (code, article) tuples for comparison
        docs_a = [(m.get("code", ""), m.get("article", "")) 
                  for m in res_a.metadatas]
        docs_b = [(m.get("code", ""), m.get("article", "")) 
                  for m in res_b.metadatas]
        
        for k in k_values:
            set_a = set(docs_a[:k])
            set_b = set(docs_b[:k])
            
            if len(set_a) > 0:
                overlap = len(set_a & set_b) / len(set_a)
                overlaps[k].append(overlap)
    
    return {k: float(np.mean(overlaps[k])) if overlaps[k] else 0.0 for k in k_values}


def calculate_rank_correlation(
    results_a: List[ModelResult],
    results_b: List[ModelResult]
) -> Tuple[float, float]:
    """Calculate Spearman and Kendall rank correlations"""
    # We need to compare rankings for common documents
    all_correlations_spearman = []
    all_correlations_kendall = []
    
    for res_a, res_b in zip(results_a, results_b):
        # Get document identifiers
        docs_a = [(m.get("code", ""), m.get("article", "")) for m in res_a.metadatas]
        docs_b = [(m.get("code", ""), m.get("article", "")) for m in res_b.metadatas]
        
        # Find common documents
        common_docs = set(docs_a) & set(docs_b)
        if len(common_docs) < 2:
            continue
        
        # Create rank vectors for common docs
        ranks_a = [docs_a.index(doc) + 1 for doc in common_docs]
        ranks_b = [docs_b.index(doc) + 1 for doc in common_docs]
        
        if len(ranks_a) >= 2:
            spearman, _ = spearmanr(ranks_a, ranks_b)
            kendall, _ = kendalltau(ranks_a, ranks_b)
            if not np.isnan(spearman):
                all_correlations_spearman.append(spearman)
            if not np.isnan(kendall):
                all_correlations_kendall.append(kendall)
    
    avg_spearman = float(np.mean(all_correlations_spearman)) if all_correlations_spearman else 0.0
    avg_kendall = float(np.mean(all_correlations_kendall)) if all_correlations_kendall else 0.0
    
    return avg_spearman, avg_kendall


def calculate_distance_distribution_stats(results: List[ModelResult]) -> Dict:
    """Calculate statistics about distance/score distributions"""
    all_distances = []
    first_to_second_ratios = []
    
    for result in results:
        if len(result.distances) >= 2:
            all_distances.extend(result.distances)
            # Ratio of best to second-best distance (higher = more confident)
            ratio = result.distances[1] / (result.distances[0] + 1e-8)
            first_to_second_ratios.append(ratio)
    
    if not all_distances:
        return {}
    
    return {
        "mean_distance": float(np.mean(all_distances)),
        "std_distance": float(np.std(all_distances)),
        "min_distance": float(np.min(all_distances)),
        "max_distance": float(np.max(all_distances)),
        "mean_first_to_second_ratio": float(np.mean(first_to_second_ratios)) if first_to_second_ratios else 0.0,
        "confidence_score": float(np.mean([1.0 / (1.0 + d) for d in all_distances]))  # Normalized confidence
    }


def calculate_semantic_coherence(
    results: List[ModelResult],
    embedding_model: EmbeddingModel,
    model_name: str = ""
) -> float:
    """
    Calculate semantic coherence: how similar are retrieved docs to query?
    Returns average cosine similarity between query and retrieved docs.
    """
    similarities = []
    total = len(results)
    
    for i, result in enumerate(results):
        if not result.retrieved_docs:
            continue
        
        # Progress indicator
        if model_name and (i + 1) % 5 == 0:
            print(f"    Calculating coherence for {model_name}: {i+1}/{total} queries...")
        
        # Embed query
        query_embedding = embedding_model.embed_query(result.query)
        
        # Embed retrieved documents
        doc_embeddings = embedding_model.embed_documents(result.retrieved_docs)
        
        # Calculate cosine similarities
        for doc_emb in doc_embeddings:
            similarity = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb) + 1e-8
            )
            similarities.append(float(similarity))
    
    return float(np.mean(similarities)) if similarities else 0.0


def compare_two_models(
    model_a_name: str,
    model_b_name: str,
    results_a: List[ModelResult],
    results_b: List[ModelResult],
    embedding_model_a: EmbeddingModel,
    embedding_model_b: EmbeddingModel
) -> ComparisonMetrics:
    """Compare two models comprehensively"""
    
    # Calculate overlaps
    overlap_at_k = calculate_overlap_at_k(results_a, results_b, k_values=[1, 3, 5])
    
    # Calculate rank correlations
    spearman, kendall = calculate_rank_correlation(results_a, results_b)
    
    # Distance correlation (how similar are the distance distributions)
    dist_stats_a = calculate_distance_distribution_stats(results_a)
    dist_stats_b = calculate_distance_distribution_stats(results_b)
    distance_corr = 0.0
    if dist_stats_a and dist_stats_b:
        distance_corr = 1.0 - abs(dist_stats_a["mean_distance"] - dist_stats_b["mean_distance"]) / max(
            dist_stats_a["mean_distance"], dist_stats_b["mean_distance"], 1e-8
        )
    
    # Semantic coherence (with progress indicators)
    print(f"  Calculating semantic coherence for {model_a_name}...")
    sem_a = calculate_semantic_coherence(results_a, embedding_model_a, model_a_name)
    print(f"  Calculating semantic coherence for {model_b_name}...")
    sem_b = calculate_semantic_coherence(results_b, embedding_model_b, model_b_name)
    
    return ComparisonMetrics(
        model_a=model_a_name,
        model_b=model_b_name,
        overlap_at_k=overlap_at_k,
        spearman_correlation=spearman,
        kendall_tau=kendall,
        distance_correlation=distance_corr,
        semantic_coherence_a=sem_a,
        semantic_coherence_b=sem_b
    )


def llm_as_judge_compare(
    query: str,
    results_a: ModelResult,
    results_b: ModelResult,
    llm_chain: LLMChainWrapper
) -> Dict:
    """
    Use LLM to judge which model's retrieval is better.
    Returns dict with winner and reasoning.
    """
    # Format context from both models
    context_a = "\n\n".join([f"[{m.get('code', 'N/A')} {m.get('article', 'N/A')}]: {doc}" 
                             for doc, m in zip(results_a.retrieved_docs, results_a.metadatas)])
    context_b = "\n\n".join([f"[{m.get('code', 'N/A')} {m.get('article', 'N/A')}]: {doc}" 
                             for doc, m in zip(results_b.retrieved_docs, results_b.metadatas)])
    
    prompt = f"""Tu es un expert en droit français. Tu dois évaluer la qualité de deux ensembles de documents juridiques retournés pour une question.

Question: {query}

Ensemble A:
{context_a}

Ensemble B:
{context_b}

Évalue selon ces critères (note sur 5):
1. Pertinence: Les documents répondent-ils directement à la question?
2. Citations: Les documents citent-ils des articles de loi spécifiques?
3. Complétude: L'ensemble couvre-t-il tous les aspects de la question?

Réponds en JSON:
{{
  "winner": "A" ou "B" ou "tie",
  "score_a": <note 1-5>,
  "score_b": <note 1-5>,
  "reasoning": "explication en français"
}}"""
    
    response = llm_chain.llm.invoke(prompt)
    content = getattr(response, "content", getattr(response, "text", str(response)))
    
    # Try to parse JSON
    json_start = content.find("{")
    json_end = content.rfind("}")
    if json_start >= 0 and json_end >= 0:
        return json.loads(content[json_start:json_end+1])
    
    raise ValueError(f"Could not parse LLM response as JSON: {content[:200]}")


def run_full_comparison(
    models: Dict[str, str],
    queries: List[Dict],
    use_llm_judge: bool = False,
    sample_size_for_judge: int = 5
) -> Dict:
    """
    Run full comparison across all models and queries.
    
    Returns comprehensive results dict.
    """
    print("\n" + "="*80)
    print("STARTING FULL EMBEDDING MODEL COMPARISON")
    print("="*80)
    print(f"Models to compare: {list(models.keys())}")
    print(f"Number of queries: {len(queries)}")
    print(f"Using LLM-as-judge: {use_llm_judge}")
    print("="*80)
    
    # Run queries on all models
    all_results = {}
    for model_name, model_path in models.items():
        results = run_queries_on_model(model_name, model_path, queries)
        if results:
            all_results[model_name] = results
    
    if len(all_results) < 2:
        print("ERROR: Need at least 2 models with successful results to compare")
        return {}
    
    # Calculate comparison metrics
    model_names = list(all_results.keys())
    comparison_matrix = {}
    
    print("\n" + "-"*80)
    print("CALCULATING CROSS-MODEL METRICS")
    print("-"*80)
    
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i+1:]:
            print(f"\nComparing {model_a} vs {model_b}...")
            
            # Load models and calculate semantic coherence
            emb_a = EmbeddingModel(model_name=models[model_a])
            emb_b = EmbeddingModel(model_name=models[model_b])
            
            metrics = compare_two_models(
                model_a, model_b,
                all_results[model_a],
                all_results[model_b],
                emb_a, emb_b
            )
            
            comparison_matrix[f"{model_a}_vs_{model_b}"] = asdict(metrics)
            
            print(f"  Overlap@1: {metrics.overlap_at_k.get(1, 0):.2%}")
            print(f"  Overlap@3: {metrics.overlap_at_k.get(3, 0):.2%}")
            print(f"  Overlap@5: {metrics.overlap_at_k.get(5, 0):.2%}")
            print(f"  Spearman: {metrics.spearman_correlation:.3f}")
            print(f"  Kendall tau: {metrics.kendall_tau:.3f}")
            
            # Clean up GPU memory after each comparison
            print(f"  Cleaning up GPU memory after comparison...")
            # Move models to CPU first, then delete (more aggressive cleanup)
            import torch
            if hasattr(emb_a, 'model') and emb_a.model is not None:
                emb_a.model = emb_a.model.to('cpu')
            if hasattr(emb_b, 'model') and emb_b.model is not None:
                emb_b.model = emb_b.model.to('cpu')
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Now safe to delete
            del emb_a
            del emb_b
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset peak memory stats to get accurate readings
            torch.cuda.reset_peak_memory_stats()
            print(f"  [OK] GPU memory cleared")
    
    # Calculate per-model statistics
    print("\n" + "-"*80)
    print("PER-MODEL STATISTICS")
    print("-"*80)
    
    model_stats = {}
    for model_name, results in all_results.items():
        stats = calculate_distance_distribution_stats(results)
        avg_time = float(np.mean([r.query_time_ms for r in results]))
        
        # Categorize by in-domain vs out-of-domain
        in_domain = [r for r in results if r.query_category not in 
                    ["inheritance", "family_law", "housing", "data_protection"]]
        out_domain = [r for r in results if r.query_category in 
                     ["inheritance", "family_law", "housing", "data_protection"]]
        
        in_domain_conf = calculate_distance_distribution_stats(in_domain) if in_domain else {}
        out_domain_conf = calculate_distance_distribution_stats(out_domain) if out_domain else {}
        
        model_stats[model_name] = {
            "avg_query_time_ms": avg_time,
            "distance_stats": stats,
            "in_domain_confidence": in_domain_conf.get("confidence_score", 0),
            "out_domain_confidence": out_domain_conf.get("confidence_score", 0),
            "num_queries": len(results)
        }
        
        print(f"\n{model_name}:")
        print(f"  Avg query time: {avg_time:.1f}ms")
        print(f"  Mean distance: {stats.get('mean_distance', 0):.2f}")
        print(f"  In-domain confidence: {in_domain_conf.get('confidence_score', 0):.3f}")
        print(f"  Out-domain confidence: {out_domain_conf.get('confidence_score', 0):.3f}")
    
    # LLM-as-judge comparison
    llm_judge_results = {}
    if use_llm_judge and len(model_names) >= 2:
        print("\n" + "-"*80)
        print("RUNNING LLM-AS-JUDGE EVALUATION")
        print("-"*80)
        
        llm_chain = LLMChainWrapper()
        
        # Sample queries for LLM judge (mix of in-domain and out-domain)
        sample_indices = np.random.choice(
            len(queries), 
            min(sample_size_for_judge, len(queries)), 
            replace=False
        )
        
        wins = defaultdict(int)
        scores = defaultdict(list)
        
        for idx in sample_indices:
            query_data = queries[idx]
            query = query_data["query"]
            
            # Compare first two models
            model_a, model_b = model_names[0], model_names[1]
            result_a = all_results[model_a][idx]
            result_b = all_results[model_b][idx]
            
            print(f"\nQuery: {query[:60]}...")
            judge_result = llm_as_judge_compare(query, result_a, result_b, llm_chain)
            
            winner = judge_result.get("winner", "error")
            wins[winner] += 1
            scores[model_a].append(judge_result.get("score_a", 0))
            scores[model_b].append(judge_result.get("score_b", 0))
            
            print(f"  Winner: {winner}")
            print(f"  {model_a}: {judge_result.get('score_a', 0)}/5")
            print(f"  {model_b}: {judge_result.get('score_b', 0)}/5")
        
        llm_judge_results = {
            "wins": dict(wins),
            "avg_scores": {model: float(np.mean(scores[model])) for model in scores},
            "sample_size": int(len(sample_indices))
        }
    
    # Compile final report
    report = {
        "timestamp": datetime.now().isoformat(),
        "num_models": len(model_names),
        "num_queries": len(queries),
        "models": model_names,
        "comparison_matrix": comparison_matrix,
        "model_statistics": model_stats,
        "llm_judge_results": llm_judge_results,
        "query_categories": list(set(q["category"] for q in queries))
    }
    
    return report


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    # Handle numpy scalar types
    if hasattr(obj, 'dtype'):  # numpy scalar
        if np.issubdtype(obj.dtype, np.integer):
            return int(obj)
        elif np.issubdtype(obj.dtype, np.floating):
            return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


def save_comparison_report(report: Dict, output_path: str = "comparison_report.json"):
    """Save comparison report to JSON file"""
    # Convert numpy types to Python native types
    serializable_report = convert_to_serializable(report)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_report, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Comparison report saved to: {output_path}")


def print_comparison_summary(report: Dict):
    """Print a readable summary of the comparison"""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    # Model stats summary
    print("\n📊 Model Performance Overview:")
    print("-" * 80)
    print(f"{'Model':<30} {'Query Time':<15} {'Confidence':<15} {'In/Out Domain':<20}")
    print("-" * 80)
    
    for model_name, stats in report["model_statistics"].items():
        in_conf = stats.get("in_domain_confidence", 0)
        out_conf = stats.get("out_domain_confidence", 0)
        print(f"{model_name:<30} "
              f"{stats['avg_query_time_ms']:<15.1f} "
              f"{stats['distance_stats'].get('confidence_score', 0):<15.3f} "
              f"{in_conf:.3f}/{out_conf:.3f}")
    
    # Cross-model agreement
    print("\n\n📈 Cross-Model Agreement (Overlap@5):")
    print("-" * 80)
    
    for comparison, metrics in report["comparison_matrix"].items():
        model_a, model_b = comparison.split("_vs_")
        overlap = metrics.get("overlap_at_k", {}).get(5, 0)
        spearman = metrics.get("spearman_correlation", 0)
        print(f"{model_a:<25} vs {model_b:<25}: {overlap:>6.1%} overlap, ρ={spearman:.3f}")
    
    # LLM Judge results
    if report.get("llm_judge_results"):
        print("\n\n🤖 LLM-as-Judge Results:")
        print("-" * 80)
        wins = report["llm_judge_results"]["wins"]
        scores = report["llm_judge_results"]["avg_scores"]
        
        for model, score in scores.items():
            win_count = wins.get(model.split("_")[0] if "_" in model else model, 0)
            print(f"{model}: {score:.2f}/5 average score, {win_count} wins")
    
    print("\n" + "="*80)
    
    # Recommendations
    print("\n💡 Recommendations:")
    print("-" * 80)
    
    # Find best model by confidence on in-domain queries
    best_confidence = max(
        report["model_statistics"].items(),
        key=lambda x: x[1].get("in_domain_confidence", 0)
    )
    print(f"• Best in-domain confidence: {best_confidence[0]} "
          f"({best_confidence[1]['in_domain_confidence']:.3f})")
    
    # Find fastest model
    fastest = min(
        report["model_statistics"].items(),
        key=lambda x: x[1]["avg_query_time_ms"]
    )
    print(f"• Fastest model: {fastest[0]} ({fastest[1]['avg_query_time_ms']:.1f}ms)")
    
    # Check for high-agreement pairs
    high_agreement = [
        (comp, metrics) for comp, metrics in report["comparison_matrix"].items()
        if metrics.get("overlap_at_k", {}).get(5, 0) > 0.7
    ]
    if high_agreement:
        print(f"• {len(high_agreement)} model pair(s) with >70% agreement "
              "(likely both good or both poor)")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare embedding models for French Legal RAG"
    )
    
    parser.add_argument(
        "--build-indexes",
        action="store_true",
        help="Build indexes for all test models"
    )
    parser.add_argument(
        "--compare-queries",
        action="store_true",
        help="Run comparison on test queries"
    )
    parser.add_argument(
        "--full-evaluation",
        action="store_true",
        help="Run full evaluation including all metrics"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Number of articles to use (default: 5000)"
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="hnsw_flat",
        choices=["flat_l2", "ivf_flat", "hnsw_flat"],
        help="FAISS index type"
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default=None,
        help="JSON file with custom test queries"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to test (space-separated)"
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Enable LLM-as-judge evaluation"
    )
    parser.add_argument(
        "--judge-sample",
        type=int,
        default=5,
        help="Number of queries to evaluate with LLM judge"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_report.json",
        help="Output file for comparison report"
    )
    
    args = parser.parse_args()
    
    load_dotenv()
    
    # Select models to test
    models_to_test = TEST_MODELS
    if args.models:
        models_to_test = {k: v for k, v in TEST_MODELS.items() if k in args.models}
    
    # Load test queries
    if args.queries_file and os.path.exists(args.queries_file):
        with open(args.queries_file, "r", encoding="utf-8") as f:
            test_queries = json.load(f)
    else:
        test_queries = DEFAULT_TEST_QUERIES
    
    print(f"Testing {len(models_to_test)} models on {len(test_queries)} queries")
    
    # Build indexes if requested
    if args.build_indexes:
        print("\n" + "="*80)
        print("BUILDING INDEXES FOR ALL MODELS")
        print("="*80)
        
        # Load and preprocess data once
        dataset = load_french_legal_data(config.DATASET_NAME)
        filtered = filter_active_articles(dataset)
        
        if os.path.exists("data/processed_articles.json"):
            print("Loading processed articles...")
            articles = load_processed_data("data/processed_articles.json")
        else:
            print("Preprocessing articles...")
            articles = preprocess_articles(filtered)
            save_processed_data(articles, "data/processed_articles.json")
        
        if args.limit and len(articles) > args.limit:
            print(f"Limiting to {args.limit} articles")
            articles = articles[:args.limit]
        
        build_results = build_all_indexes(models_to_test, articles, args.index_type)
        
        # Save build results
        with open("index_build_results.json", "w", encoding="utf-8") as f:
            json.dump(build_results, f, ensure_ascii=False, indent=2)
        
        print("\n✓ Index building complete!")
    
    # Run comparison
    if args.compare_queries or args.full_evaluation:
        report = run_full_comparison(
            models_to_test,
            test_queries,
            use_llm_judge=args.llm_judge,
            sample_size_for_judge=args.judge_sample
        )
        
        if report:
            save_comparison_report(report, args.output)
            print_comparison_summary(report)
    
    if not any([args.build_indexes, args.compare_queries, args.full_evaluation]):
        print("\nUsage:")
        print("  python compare_embeddings.py --build-indexes --limit 5000")
        print("  python compare_embeddings.py --full-evaluation --llm-judge")
        print("  python compare_embeddings.py --compare-queries --models sentence-camembert-base voyage-4")
        print("\nFor more options: python compare_embeddings.py --help")


if __name__ == "__main__":
    main()
