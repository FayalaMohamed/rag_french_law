"""
Retrieval Strategy Comparison Script for French Legal RAG

This script systematically compares different retrieval strategies:
- HyDE (Hypothetical Document Embeddings): on/off
- Query Decomposition: on/off  
- Variable top_k: different values

Usage:
    python compare_retrieval_strategies.py --run-comparison --queries-file test_queries.json
    python compare_retrieval_strategies.py --full-evaluation --output strategy_comparison.json

The script produces detailed metrics and visual comparisons between strategies.
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from dotenv import load_dotenv

from data.loader import load_french_legal_data, filter_active_articles
from models.embeddings import EmbeddingModel
from data.vector_store import FAISSVectorStore
from chains.rag_chain import RAGPipeline
from chains.llm_chain import LLMChainWrapper
from utils.evaluation import RAGEvaluator, RAGEvaluationResult
from utils.query_classifier import QueryClassifier
import config


@dataclass
class StrategyConfig:
    """Configuration for a retrieval strategy"""
    name: str
    use_hyde: bool
    use_decomposition: bool
    k: int
    description: str


@dataclass
class StrategyResult:
    """Results for a single strategy on a single query"""
    strategy_name: str
    query: str
    query_type: str
    query_category: str
    retrieved_docs: List[str]
    distances: List[float]
    metadatas: List[Dict]
    answer: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    num_sub_queries: int = 0


DEFAULT_STRATEGIES = [
    StrategyConfig(
        name="baseline",
        use_hyde=False,
        use_decomposition=False,
        k=5,
        description="Simple retrieval without enhancements"
    ),
    StrategyConfig(
        name="hyde_only",
        use_hyde=True,
        use_decomposition=False,
        k=5,
        description="HyDE enabled, no decomposition"
    ),
    StrategyConfig(
        name="decomposition_only",
        use_hyde=False,
        use_decomposition=True,
        k=5,
        description="Query decomposition only"
    ),
    StrategyConfig(
        name="hyde_and_decomposition",
        use_hyde=True,
        use_decomposition=True,
        k=5,
        description="Both HyDE and decomposition enabled"
    ),
    StrategyConfig(
        name="high_recall_k10",
        use_hyde=True,
        use_decomposition=True,
        k=10,
        description="HyDE + decomposition with k=10"
    ),
    StrategyConfig(
        name="precision_focused_k3",
        use_hyde=True,
        use_decomposition=False,
        k=3,
        description="HyDE only with k=3 for precision"
    ),
]

DEFAULT_TEST_QUERIES = [
    # In-context queries (grounded in indexed documents - 5 queries)
    {
        "query": "Depuis la loi du 28 décembre 2015, quels sont les impacts concrets sur les droits des personnes âgées de plus de 60 ans concernant les prestations sociales liées à la dépendance, comme les aides financières ou les services d'accompagnement, selon les articles modifiés du Code de la sécurité sociale (L115-2-1 et L115-9) ?",
        "category": "rights",
        "expected_aspects": [
            "modification des critères d'éligibilité ou de financement des aides",
            "impact sur les droits aux prestations liées à la dépendance ou à l'autonomie",
            "référence aux articles L115-2-1 et L115-9 du Code de la sécurité sociale"
        ],
        "source_code": "LOI n° 2015-1776 du 28 décembre 2015 relative à l'adaptation de la société au vieillissement (1)",
        "source_article": "6",
        "in_context": True
    },
    {
        "query": "Si je vends un bien en France et que le contrat de cession est signé le 15 mars 2026, dois-je respecter un délai de deux mois à compter de cette date pour effectuer une certaine obligation légale (par exemple, une déclaration ou un paiement lié à la cession) ?",
        "category": "temporal",
        "expected_aspects": [
            "définition de la date de cession (date de conclusion du contrat)",
            "application du délai de deux mois à partir de cette date",
            "interprétation du premier alinéa de l'article L. 23-10-1 du Code de commerce"
        ],
        "source_code": "Code de commerce",
        "source_article": "D23-10-1",
        "in_context": True
    },
    {
        "query": "Dans le cadre du Code de la sécurité sociale, selon l'article L851-1 modifié par la loi de finances rectificative de 2015, quelles sont les règles spécifiques qui encadrent les droits des travailleurs en matière de protection sociale pour les employeurs ou les salariés concernés par cette disposition ?",
        "category": "rights",
        "expected_aspects": [
            "impact de la modification légale sur les règles générales de protection sociale",
            "référence aux conditions d'application de l'article L851-1",
            "contexte de la loi de finances rectificative de 2015 (art. 118)"
        ],
        "source_code": "LOI n° 2015-1786 du 29 décembre 2015 de finances rectificative pour 2015 (1)",
        "source_article": "118",
        "in_context": True
    },
    {
        "query": "Dans le cadre d'une entreprise adaptée, quelles sont les obligations légales précises pour l'agrément de l'entreprise selon le décret 2018-1334, et comment cela impacte-t-il les conditions de financement et d'accompagnement des salariés en situation de handicap ?",
        "category": "conditional",
        "expected_aspects": [
            "conditions d'agrément (art. R5212-5 et autres)",
            "modalités de financement (art. D5212-22 et références connexes)",
            "accompagnement spécifique des salariés (art. R5213-46-2, R5213-70, R5213-71, R5213-73, R5523-2)"
        ],
        "source_code": "Décret n° 2018-1334 du 28 décembre 2018 relatif aux conditions d'agrément et de financement des entreprises adaptées ainsi qu'aux modalités d'accompagnement spécifique de leurs salariés en situation de handicap",
        "source_article": "2",
        "in_context": True
    },
    {
        "query": "Si mon employeur me fait une saisie sur mon salaire pour rembourser une dette, selon le décret de 2015, quelle partie exacte de ma rémunération peut être saisie et sous quelles conditions précises (montant maximal, durée maximale, etc.) ?",
        "category": "conditional",
        "expected_aspects": [
            "montant maximal de la saisie sur salaire",
            "durée maximale de la saisie",
            "référence au barème spécifique mentionné dans le décret (Art. R3252-2 du Code du travail)"
        ],
        "source_code": "Décret n° 2015-1842 du 30 décembre 2015 révisant le barème des saisies et cessions des rémunérations",
        "source_article": "1",
        "in_context": True
    },
    # Out-of-context queries (NOT in indexed documents - 3 queries)
    {
        "query": "Quelles sont les règles de l'impôt sur le revenu en France?",
        "category": "out_of_context",
        "expected_domain": "droit immobilier",
        "in_context": False
    },
    {
        "query": "Comment déposer un brevet d'invention?",
        "category": "out_of_context",
        "expected_domain": "droit international",
        "in_context": False
    },
    {
        "query": "Quels sont les droits du consommateur en cas de litige?",
        "category": "out_of_context",
        "expected_domain": "droit de la consommation",
        "in_context": False
    }
]


def run_strategy_on_query(
    strategy: StrategyConfig,
    query: str,
    pipeline: RAGPipeline,
    query_classifier: QueryClassifier
) -> StrategyResult:
    """
    Run a single strategy on a single query.
    
    Returns:
        StrategyResult with timing and results
    """
    # Classify query
    classification = query_classifier.classify(query)
    query_type = classification["query_type"]
    
    retrieval_start = time.time()
    
    if strategy.use_decomposition:
        # Decomposition retrieves multiple times
        sub_queries = pipeline.decompose_query(query)
        num_sub_queries = len(sub_queries)
        
        all_docs = []
        all_distances = []
        all_metadatas = []
        
        for sub_query in sub_queries:
            docs, dists, metas = pipeline.retrieve(
                sub_query,
                k=strategy.k,
                use_hyde=strategy.use_hyde
            )
            all_docs.extend(docs)
            all_distances.extend(dists)
            all_metadatas.extend(metas)
        
        # Deduplicate and sort
        unique_results = []
        seen_texts = set()
        for doc, dist, meta in zip(all_docs, all_distances, all_metadatas):
            if doc not in seen_texts:
                seen_texts.add(doc)
                unique_results.append((doc, dist, meta))
        
        unique_results.sort(key=lambda x: x[1])
        
        retrieved_docs = [r[0] for r in unique_results[:strategy.k]]
        distances = [r[1] for r in unique_results[:strategy.k]]
        metadatas = [r[2] for r in unique_results[:strategy.k]]
    else:
        # Simple retrieval
        num_sub_queries = 1
        retrieved_docs, distances, metadatas = pipeline.retrieve(
            query,
            k=strategy.k,
            use_hyde=strategy.use_hyde
        )
    
    retrieval_time = (time.time() - retrieval_start) * 1000
    
    # Time the generation
    generation_start = time.time()
    context = "\n\n".join(retrieved_docs)
    answer = pipeline.generate(query, context)
    generation_time = (time.time() - generation_start) * 1000
    
    total_time = retrieval_time + generation_time
    
    return StrategyResult(
        strategy_name=strategy.name,
        query=query,
        query_type=query_type,
        query_category=classification["domain"],
        retrieved_docs=retrieved_docs,
        distances=distances,
        metadatas=metadatas,
        answer=answer,
        retrieval_time_ms=retrieval_time,
        generation_time_ms=generation_time,
        total_time_ms=total_time,
        num_sub_queries=num_sub_queries
    )


def run_all_strategies_on_queries(
    strategies: List[StrategyConfig],
    queries: List[Dict],
    pipeline: RAGPipeline,
    query_classifier: QueryClassifier
) -> Dict[str, List[StrategyResult]]:
    """
    Run all strategies on all queries.
    
    Returns:
        Dictionary mapping strategy names to lists of results
    """
    all_results = {strategy.name: [] for strategy in strategies}
    
    total_queries = len(queries)
    total_strategies = len(strategies)
    total_runs = total_queries * total_strategies
    
    print(f"\nRunning {total_strategies} strategies on {total_queries} queries ({total_runs} total runs)")
    print("=" * 80)
    
    run_count = 0
    for query_idx, query_data in enumerate(queries):
        query = query_data["query"]
        print(f"\n[{query_idx + 1}/{total_queries}] Query: {query[:60]}...")
        
        for strategy in strategies:
            run_count += 1
            print(f"  [{run_count}/{total_runs}] Strategy: {strategy.name}...", end=" ")
            
            try:
                result = run_strategy_on_query(
                    strategy, query, pipeline, query_classifier
                )
                all_results[strategy.name].append(result)
                print(f"✓ ({result.total_time_ms:.0f}ms)")
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
    
    return all_results


def llm_judge_answer_quality(
    query: str,
    answer: str,
    retrieved_docs: List[str],
    metadatas: List[Dict],
    llm_chain: LLMChainWrapper
) -> Dict:
    """
    Use LLM to judge answer quality on multiple dimensions.
    Returns scores and reasoning.
    """
    # Format sources
    sources_text = "\n\n".join([
        f"[{m.get('code', 'N/A')} {m.get('article', 'N/A')}]: {doc[:300]}..."
        for doc, m in zip(retrieved_docs[:3], metadatas[:3])  # Top 3 sources
    ])
    
    prompt = f"""Tu es un expert en droit français. Évalue la qualité de cette réponse juridique.

Question: {query}

Réponse générée:
{answer}

Sources utilisées:
{sources_text}

Évalue la réponse selon ces critères (note 1-5 pour chaque):

1. **Exactitude juridique**: La réponse est-elle factuellement correcte selon le droit français?
2. **Complétude**: La réponse couvre-t-elle tous les aspects importants de la question?
3. **Pertinence**: La réponse répond-elle directement à la question posée?
4. **Citations**: La réponse cite-t-elle correctement les articles de loi?
5. **Clarté**: La réponse est-elle claire et bien structurée?

Réponds UNIQUEMENT en JSON valide:
{{
  "exactitude": <1-5>,
  "completude": <1-5>,
  "pertinence": <1-5>,
  "citations": <1-5>,
  "clarte": <1-5>,
  "score_total": <moyenne des 5>,
  "factual_errors": ["liste des erreurs factuelles si détectées"],
  "strengths": ["points forts"],
  "weaknesses": ["points faibles"]
}}"""
    
    try:
        response = llm_chain.llm.invoke(prompt)
        content = getattr(response, "content", getattr(response, "text", str(response)))
        
        # Clean control characters from content
        content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')
        
        # Remove markdown code block markers if present
        content = content.replace("```json", "").replace("```", "")
        
        # Try to find and parse JSON
        json_start = content.find("{")
        json_end = content.rfind("}")
        if json_start >= 0 and json_end >= 0:
            json_str = content[json_start:json_end+1]
            # Remove any problematic escape sequences
            for i in range(32):
                if i not in [9, 10, 13]:  # Keep tab, newline, carriage return
                    json_str = json_str.replace(chr(i), '')
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parsing failed: {e}")
    except Exception as e:
        print(f"LLM judge error: {e}")
    
    # Return default if parsing fails
    return {
        "exactitude": 3,
        "completude": 3,
        "pertinence": 3,
        "citations": 3,
        "clarte": 3,
        "score_total": 3.0,
        "factual_errors": [],
        "strengths": [],
        "weaknesses": ["Erreur d'évaluation"]
    }


def calculate_strategy_metrics(
    results: List[StrategyResult],
    evaluator: RAGEvaluator,
    llm_chain: LLMChainWrapper = None,
    use_llm_judge: bool = False,
    llm_judge_sample: int = 3
) -> Dict:
    """
    Calculate metrics for a strategy based on its results.
    """
    if not results:
        return {}
    
    avg_retrieval_time = np.mean([r.retrieval_time_ms for r in results])
    avg_generation_time = np.mean([r.generation_time_ms for r in results])
    avg_total_time = np.mean([r.total_time_ms for r in results])
    
    # Generation quality metrics (using evaluator)
    gen_metrics_list = []
    for result in results:
        gen_metrics = evaluator.generation_evaluator.evaluate_generation(
            question=result.query,
            answer=result.answer,
            retrieved_docs=[{"text": doc, "metadata": meta} 
                          for doc, meta in zip(result.retrieved_docs, result.metadatas)]
        )
        gen_metrics_list.append(gen_metrics)
    
    # Average generation metrics
    avg_gen_metrics = {
        "answer_relevance": float(np.mean([m.answer_relevance_score for m in gen_metrics_list])),
        "citation_precision": float(np.mean([m.citation_precision for m in gen_metrics_list])),
        "citation_recall": float(np.mean([m.citation_recall for m in gen_metrics_list])),
        "source_groundedness": float(np.mean([m.source_groundedness for m in gen_metrics_list])),
        "completeness": float(np.mean([m.completeness_score for m in gen_metrics_list])),
    }
    
    # LLM-as-judge evaluation (sample subset to save time/cost)
    llm_judge_scores = []
    if use_llm_judge and llm_chain and len(results) > 0:
        print(f"    Running LLM judge on {min(llm_judge_sample, len(results))} samples...")
        
        # Deterministic sampling: prioritize diverse query types and in/out of context
        # Group results by query type and in_context status
        in_context_indices = [i for i, r in enumerate(results) if getattr(r, 'in_context', True)]
        out_context_indices = [i for i, r in enumerate(results) if not getattr(r, 'in_context', False)]
        
        sample_indices = []
        
        # Always include at least one out-of-context query if available
        if out_context_indices and llm_judge_sample >= 1:
            sample_indices.append(out_context_indices[0])
        
        # Add in-context queries to fill remaining slots
        remaining_slots = llm_judge_sample - len(sample_indices)
        if remaining_slots > 0 and in_context_indices:
            # Take first N in-context queries (deterministic)
            sample_indices.extend(in_context_indices[:remaining_slots])
        
        # If we still need more, add from remaining queries
        remaining_slots = llm_judge_sample - len(sample_indices)
        if remaining_slots > 0:
            all_indices = list(range(len(results)))
            for idx in all_indices:
                if idx not in sample_indices:
                    sample_indices.append(idx)
                    remaining_slots -= 1
                    if remaining_slots <= 0:
                        break
        
        sample_indices = sample_indices[:llm_judge_sample]  # Ensure we don't exceed
        
        print(f"      Selected queries: {sample_indices}")
        if out_context_indices:
            print(f"      (Includes out-of-context: {any(i in out_context_indices for i in sample_indices)})")
        
        for idx in sample_indices:
            result = results[idx]
            judge_result = llm_judge_answer_quality(
                result.query,
                result.answer,
                result.retrieved_docs,
                result.metadatas,
                llm_chain
            )
            llm_judge_scores.append(judge_result)
    
    # Aggregate LLM judge scores
    llm_judge_metrics = {}
    if llm_judge_scores:
        # Use .get() with defaults to handle missing keys
        llm_judge_metrics = {
            "exactitude": float(np.mean([s.get("exactitude", s.get("accuracy", 3)) for s in llm_judge_scores])),
            "completude": float(np.mean([s.get("completude", s.get("completeness", 3)) for s in llm_judge_scores])),
            "pertinence": float(np.mean([s.get("pertinence", s.get("relevance", 3)) for s in llm_judge_scores])),
            "citations": float(np.mean([s.get("citations", s.get("citation_quality", 3)) for s in llm_judge_scores])),
            "clarte": float(np.mean([s.get("clarte", s.get("clarity", 3)) for s in llm_judge_scores])),
            "score_total": float(np.mean([s.get("score_total", s.get("total_score", 3)) for s in llm_judge_scores])),
            "num_judged": len(llm_judge_scores),
        }
    
    # Decomposition effectiveness
    if any(r.num_sub_queries > 1 for r in results):
        avg_sub_queries = np.mean([r.num_sub_queries for r in results if r.num_sub_queries > 1])
    else:
        avg_sub_queries = 1.0
    
    # Query type breakdown
    query_type_times = {}
    query_type_quality = {}
    for i, result in enumerate(results):
        qtype = result.query_type
        if qtype not in query_type_times:
            query_type_times[qtype] = []
            query_type_quality[qtype] = []
        query_type_times[qtype].append(result.total_time_ms)
        
        # Get corresponding generation metrics by index (same order as results)
        if i < len(gen_metrics_list):
            query_type_quality[qtype].append(gen_metrics_list[i].answer_relevance_score)
    
    query_type_avg_times = {k: np.mean(v) for k, v in query_type_times.items()}
    
    # Distance statistics (confidence)
    all_distances = []
    for result in results:
        all_distances.extend(result.distances)
    
    distance_stats = {
        "mean": float(np.mean(all_distances)) if all_distances else 0,
        "std": float(np.std(all_distances)) if all_distances else 0,
        "min": float(np.min(all_distances)) if all_distances else 0,
        "max": float(np.max(all_distances)) if all_distances else 0,
    }
    
    metrics = {
        "num_queries": len(results),
        "timing": {
            "avg_retrieval_ms": float(avg_retrieval_time),
            "avg_generation_ms": float(avg_generation_time),
            "avg_total_ms": float(avg_total_time),
        },
        "quality": avg_gen_metrics,
        "llm_judge": llm_judge_metrics,
        "decomposition": {
            "avg_sub_queries": float(avg_sub_queries),
        },
        "query_type_performance": query_type_avg_times,
        "distance_stats": distance_stats,
    }
    
    return metrics


def compare_strategies(
    baseline_results: List[StrategyResult],
    new_results: List[StrategyResult],
    baseline_name: str,
    new_name: str
) -> Dict:
    """
    Compare two strategies and calculate improvement/degradation metrics.
    """
    if not baseline_results or not new_results:
        return {}
    
    comparisons = []
    
    for base, new in zip(baseline_results, new_results):
        # Time comparison
        time_diff = new.total_time_ms - base.total_time_ms
        time_pct = (time_diff / base.total_time_ms * 100) if base.total_time_ms > 0 else 0
        
        # Document overlap (measure of retrieval similarity)
        base_docs = set(d[:100] for d in base.retrieved_docs)  # First 100 chars as ID
        new_docs = set(d[:100] for d in new.retrieved_docs)
        
        if len(base_docs) > 0:
            overlap = len(base_docs & new_docs) / len(base_docs)
        else:
            overlap = 0.0
        
        comparisons.append({
            "query": base.query,
            "time_diff_ms": float(time_diff),
            "time_diff_pct": float(time_pct),
            "doc_overlap": float(overlap),
            "base_time_ms": float(base.total_time_ms),
            "new_time_ms": float(new.total_time_ms),
        })
    
    # Aggregate
    avg_time_diff = np.mean([c["time_diff_ms"] for c in comparisons])
    avg_time_pct = np.mean([c["time_diff_pct"] for c in comparisons])
    avg_overlap = np.mean([c["doc_overlap"] for c in comparisons])
    
    return {
        "baseline": baseline_name,
        "comparison": new_name,
        "avg_time_increase_ms": float(avg_time_diff),
        "avg_time_increase_pct": float(avg_time_pct),
        "avg_document_overlap": float(avg_overlap),
        "per_query_comparisons": comparisons,
    }


def run_full_strategy_comparison(
    strategies: List[StrategyConfig],
    queries: List[Dict],
    embedding_model_name: str = None,
    faiss_index_path: str = None,
    use_llm_judge: bool = False,
    llm_judge_sample: int = 3
) -> Tuple[Dict, Dict[str, List[StrategyResult]]]:
    """
    Run complete comparison of all strategies.
    
    Returns:
        Comprehensive report with all metrics and comparisons
    """
    print("\n" + "=" * 80)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("=" * 80)
    print(f"Strategies to test: {[s.name for s in strategies]}")
    print(f"Number of queries: {len(queries)}")
    print("=" * 80)
    
    # Initialize components
    print("\nInitializing components...")
    
    embedding_model = EmbeddingModel(
        model_name=embedding_model_name or config.EMBEDDING_MODEL_NAME
    )
    
    vector_store = FAISSVectorStore(
        embedding_dim=embedding_model.embedding_dim,
        index_path=faiss_index_path or config.FAISS_INDEX_PATH
    )
    
    llm_chain = LLMChainWrapper()
    pipeline = RAGPipeline(embedding_model, vector_store, llm_chain)
    query_classifier = QueryClassifier()
    evaluator = RAGEvaluator()
    
    print("✓ Components initialized")
    
    # Run all strategies
    all_results = run_all_strategies_on_queries(
        strategies, queries, pipeline, query_classifier
    )
    
    # Calculate metrics for each strategy
    print("\n" + "=" * 80)
    print("CALCULATING METRICS")
    if use_llm_judge:
        print(f"(Including LLM-as-judge on {llm_judge_sample} samples per strategy)")
    print("=" * 80)
    
    strategy_metrics = {}
    for strategy_name, results in all_results.items():
        print(f"\nCalculating metrics for {strategy_name}...")
        metrics = calculate_strategy_metrics(
            results, evaluator, 
            llm_chain=llm_chain if use_llm_judge else None,
            use_llm_judge=use_llm_judge,
            llm_judge_sample=llm_judge_sample
        )
        strategy_metrics[strategy_name] = metrics
        
        print(f"  Avg time: {metrics['timing']['avg_total_ms']:.1f}ms")
        print(f"  Auto relevance: {metrics['quality']['answer_relevance']:.3f}")
        if use_llm_judge and metrics.get('llm_judge'):
            print(f"  LLM judge score: {metrics['llm_judge']['score_total']:.2f}/5")
        print(f"  Citation precision: {metrics['quality']['citation_precision']:.3f}")
    
    # Compare each strategy to baseline
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISONS")
    print("=" * 80)
    
    comparisons = {}
    baseline_name = "baseline"
    if baseline_name in all_results:
        baseline_results = all_results[baseline_name]
        
        for strategy_name in all_results:
            if strategy_name != baseline_name:
                print(f"\nComparing {baseline_name} vs {strategy_name}...")
                comparison = compare_strategies(
                    baseline_results,
                    all_results[strategy_name],
                    baseline_name,
                    strategy_name
                )
                comparisons[f"{baseline_name}_vs_{strategy_name}"] = comparison
                
                print(f"  Time delta: {comparison['avg_time_increase_ms']:+.1f}ms "
                      f"({comparison['avg_time_increase_pct']:+.1f}%)")
                print(f"  Doc overlap: {comparison['avg_document_overlap']:.1%}")
    
    # Compile report
    report = {
        "timestamp": datetime.now().isoformat(),
        "num_strategies": len(strategies),
        "num_queries": len(queries),
        "strategies": [
            {
                "name": s.name,
                "use_hyde": s.use_hyde,
                "use_decomposition": s.use_decomposition,
                "k": s.k,
                "description": s.description
            }
            for s in strategies
        ],
        "strategy_metrics": strategy_metrics,
        "comparisons": comparisons,
    }
    
    return report, all_results


def save_report(report: Dict, output_path: str):
    """Save comparison report to JSON file"""
    def convert_to_serializable(obj):
        if hasattr(obj, 'dtype'):
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
    
    serializable_report = convert_to_serializable(report)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Report saved to: {output_path}")


def print_comparison_summary(report: Dict):
    """Print a readable summary of the comparison"""
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 80)
    
    # Strategy overview
    print("\n📋 Strategies Tested:")
    print("-" * 80)
    for strategy in report["strategies"]:
        print(f"\n{strategy['name']}:")
        print(f"  HyDE: {strategy['use_hyde']}")
        print(f"  Decomposition: {strategy['use_decomposition']}")
        print(f"  k: {strategy['k']}")
        print(f"  {strategy['description']}")
    
    # Performance metrics
    print("\n\n📊 Performance Metrics:")
    print("-" * 80)
    
    # Check if LLM judge was used
    has_llm_judge = any(
        metrics.get("llm_judge") and metrics["llm_judge"].get("score_total")
        for metrics in report["strategy_metrics"].values()
    )
    
    if has_llm_judge:
        print(f"{'Strategy':<25} {'Time':<10} {'Auto Rel':<10} {'LLM Score':<12} {'Citations':<10}")
    else:
        print(f"{'Strategy':<25} {'Avg Time':<12} {'Relevance':<12} {'Citation P':<12} {'Grounded':<12}")
    print("-" * 80)
    
    for strategy_name, metrics in report["strategy_metrics"].items():
        timing = metrics["timing"]
        quality = metrics["quality"]
        
        if has_llm_judge and metrics.get("llm_judge"):
            llm_score = metrics["llm_judge"].get("score_total", 0)
            print(f"{strategy_name:<25} "
                  f"{timing['avg_total_ms']:<10.1f} "
                  f"{quality['answer_relevance']:<10.3f} "
                  f"{llm_score:<12.2f} "
                  f"{quality['citation_precision']:<10.3f}")
        else:
            print(f"{strategy_name:<25} "
                  f"{timing['avg_total_ms']:<12.1f} "
                  f"{quality['answer_relevance']:<12.3f} "
                  f"{quality['citation_precision']:<12.3f} "
                  f"{quality['source_groundedness']:<12.3f}")
    
    # Comparisons to baseline
    if report.get("comparisons"):
        print("\n\n📈 Comparisons to Baseline:")
        print("-" * 80)
        print(f"{'Comparison':<35} {'Time Δ':<15} {'Doc Overlap':<15}")
        print("-" * 80)
        
        for comp_name, comp_data in report["comparisons"].items():
            time_delta = comp_data["avg_time_increase_ms"]
            time_pct = comp_data["avg_time_increase_pct"]
            overlap = comp_data["avg_document_overlap"]
            
            print(f"{comp_name:<35} "
                  f"{time_delta:+.1f}ms ({time_pct:+.1f}%)  "
                  f"{overlap:.1%}")
    
    # Query type performance
    print("\n\n🎯 Performance by Query Type:")
    print("-" * 80)
    
    # Collect all query types
    all_query_types = set()
    for metrics in report["strategy_metrics"].values():
        all_query_types.update(metrics.get("query_type_performance", {}).keys())
    
    if all_query_types:
        query_types = sorted(all_query_types)
        header = f"{'Strategy':<25}"
        for qt in query_types:
            header += f" {qt[:10]:<10}"
        print(header)
        print("-" * 80)
        
        for strategy_name, metrics in report["strategy_metrics"].items():
            line = f"{strategy_name:<25}"
            for qt in query_types:
                time_val = metrics.get("query_type_performance", {}).get(qt, 0)
                line += f" {time_val:<10.1f}"
            print(line)
    
    print("\n" + "=" * 80)
    
    # Recommendations
    print("\n💡 Recommendations:")
    print("-" * 80)
    
    # Find fastest strategy
    fastest = min(
        report["strategy_metrics"].items(),
        key=lambda x: x[1]["timing"]["avg_total_ms"]
    )
    print(f"• Fastest: {fastest[0]} ({fastest[1]['timing']['avg_total_ms']:.1f}ms)")
    
    # Find highest quality (use LLM judge if available, otherwise auto)
    if has_llm_judge:
        best_quality = max(
            report["strategy_metrics"].items(),
            key=lambda x: x[1].get("llm_judge", {}).get("score_total", 0)
        )
        llm_score = best_quality[1].get("llm_judge", {}).get("score_total", 0)
        print(f"• Best quality (LLM judged): {best_quality[0]} (score: {llm_score:.2f}/5)")
    else:
        best_quality = max(
            report["strategy_metrics"].items(),
            key=lambda x: x[1]["quality"]["answer_relevance"]
        )
        print(f"• Best quality (auto): {best_quality[0]} "
              f"(relevance: {best_quality[1]['quality']['answer_relevance']:.3f})")
    
    # Best citation precision
    best_citation = max(
        report["strategy_metrics"].items(),
        key=lambda x: x[1]["quality"]["citation_precision"]
    )
    print(f"• Best citations: {best_citation[0]} "
          f"(precision: {best_citation[1]['quality']['citation_precision']:.3f})")
    
    # Best trade-off (quality/time)
    print(f"\n• Quality/Speed trade-off analysis:")
    for strategy_name, metrics in report["strategy_metrics"].items():
        time_ms = metrics["timing"]["avg_total_ms"]
        if has_llm_judge and metrics.get("llm_judge"):
            quality_score = metrics["llm_judge"].get("score_total", 0)
            ratio = quality_score / (time_ms / 1000) if time_ms > 0 else 0
            print(f"  - {strategy_name}: {ratio:.2f} quality points per second")
        else:
            quality_score = metrics["quality"]["answer_relevance"]
            ratio = quality_score / (time_ms / 1000) if time_ms > 0 else 0
            print(f"  - {strategy_name}: {ratio:.3f} relevance per second")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare retrieval strategies for French Legal RAG"
    )
    
    parser.add_argument(
        "--run-comparison",
        action="store_true",
        help="Run the full strategy comparison"
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default=None,
        help="JSON file with custom test queries"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model to use (default: from config)"
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        help="Path to FAISS index (default: from config)"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=None,
        help="Specific strategies to test (space-separated names)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results/strategy_comparison_report.json",
        help="Output file for comparison report (default: benchmark_results/strategy_comparison_report.json)"
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Enable LLM-as-judge evaluation (slower but more accurate)"
    )
    parser.add_argument(
        "--llm-judge-sample",
        type=int,
        default=3,
        help="Number of queries to evaluate with LLM judge per strategy (default: 3)"
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        help="Limit to first N queries for quick testing (default: all queries)"
    )
    
    args = parser.parse_args()
    
    load_dotenv()
    
    # Load test queries
    GENERATED_QUERIES_FILE = "benchmark_results/test_queries_generated.json"
    
    if args.queries_file and os.path.exists(args.queries_file):
        # User specified a custom queries file
        with open(args.queries_file, "r", encoding="utf-8") as f:
            test_queries_data = json.load(f)
            # Handle both formats: direct list or {queries: [...]} structure
            if isinstance(test_queries_data, list):
                test_queries = test_queries_data
            else:
                test_queries = test_queries_data.get("queries", test_queries_data)
        print(f"Loaded {len(test_queries)} queries from {args.queries_file}")
    elif os.path.exists(GENERATED_QUERIES_FILE):
        # Use generated queries if available
        print(f"Loading generated queries from {GENERATED_QUERIES_FILE}...")
        with open(GENERATED_QUERIES_FILE, "r", encoding="utf-8") as f:
            test_queries_data = json.load(f)
            test_queries = test_queries_data.get("queries", [])
        in_count = sum(1 for q in test_queries if q.get("in_context") == True)
        out_count = sum(1 for q in test_queries if q.get("in_context") == False)
        print(f"✓ Loaded {len(test_queries)} queries ({in_count} in-context, {out_count} out-of-context)")
    else:
        # Fall back to default queries
        print("Generated queries not found. Using default test queries.")
        print(f"(Generate queries with: python generate_test_queries.py)")
        test_queries = DEFAULT_TEST_QUERIES
    
    # Limit queries if requested
    if args.limit_queries:
        test_queries = test_queries[:args.limit_queries]
        print(f"Limited to first {len(test_queries)} queries")
    
    # Select strategies
    strategies = DEFAULT_STRATEGIES
    if args.strategies:
        strategies = [s for s in DEFAULT_STRATEGIES if s.name in args.strategies]
    
    if not strategies:
        print("ERROR: No strategies selected!")
        return
    
    print(f"Testing {len(strategies)} strategies on {len(test_queries)} queries")
    
    # Run comparison
    if args.run_comparison:
        report, _ = run_full_strategy_comparison(
            strategies=strategies,
            queries=test_queries,
            embedding_model_name=args.embedding_model,
            faiss_index_path=args.index_path,
            use_llm_judge=args.llm_judge,
            llm_judge_sample=args.llm_judge_sample
        )
        
        save_report(report, args.output)
        print_comparison_summary(report)
    else:
        print("\nUsage:")
        print("  python compare_retrieval_strategies.py --run-comparison")
        print("  python compare_retrieval_strategies.py --run-comparison --limit-queries 1")
        print("  python compare_retrieval_strategies.py --run-comparison --llm-judge")
        print("  python compare_retrieval_strategies.py --run-comparison --strategies baseline hyde_only")
        print("  python compare_retrieval_strategies.py --run-comparison --queries-file my_queries.json")
        print("\nFor more options: python compare_retrieval_strategies.py --help")


if __name__ == "__main__":
    main()
