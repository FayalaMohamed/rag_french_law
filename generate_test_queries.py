"""
Test Query Generator for French Legal RAG

Generates 8 test questions using Ollama and the indexed documents:
- 5 in-context questions (grounded in indexed documents)
- 3 out-of-context questions (NOT in indexed documents)

Usage:
    python generate_test_queries.py --output test_queries_generated.json
    python generate_test_queries.py --num-in 5 --num-out 3 --sample-docs 20

Requirements:
    - FAISS index must be built (run main.py --build first)
    - Ollama must be running
"""

import os
import sys
import json
import argparse
import random
from typing import List, Dict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.vector_store import FAISSVectorStore
from models.embeddings import EmbeddingModel
from chains.llm_chain import LLMChainWrapper
from utils.text_processing import LegalTextProcessor
import config


def sample_documents_from_index(
    vector_store: FAISSVectorStore, 
    embedding_model: EmbeddingModel,
    num_samples: int = 20
) -> List[Dict]:
    """
    Sample random documents from the FAISS index.
    
    Returns list of documents with their metadata.
    """
    total_docs = vector_store.index.ntotal
    print(f"Index contains {total_docs} documents")
    
    sample_indices = random.sample(range(total_docs), min(num_samples, total_docs))
    
    sampled_docs = []
    
    print(f"Sampling {len(sample_indices)} documents...")
    
    varied_queries = [
        "période d'essai",
        "contrat de travail",
        "licenciement",
        "Code du travail",
        "salaire",
        "congés payés",
        "durée du travail",
        "sécurité",
        "accident",
        "représentant",
    ]
    
    all_retrieved = []
    
    for query in varied_queries: 
        docs, distances, metadatas = vector_store.search_by_text(
            query, embedding_model, k=5
        )
        for doc, meta, dist in zip(docs, metadatas, distances):
            all_retrieved.append({
                "text": doc,
                "metadata": meta,
                "distance": float(dist)
            })
    
    seen_texts = set()
    unique_docs = []
    for item in all_retrieved:
        text_preview = item["text"][:100]
        if text_preview not in seen_texts:
            seen_texts.add(text_preview)
            unique_docs.append(item)
    
    if len(unique_docs) > num_samples:
        sampled = random.sample(unique_docs, num_samples)
    else:
        sampled = unique_docs
    
    print(f"✓ Sampled {len(sampled)} unique documents")
    return sampled


def generate_in_context_questions(
    documents: List[Dict],
    llm_chain: LLMChainWrapper,
    num_questions: int = 5
) -> List[Dict]:
    """
    Generate questions that ARE answerable from the sampled documents.
    """
    questions = []
    
    print(f"\nGenerating {num_questions} in-context questions...")
    
    # Select documents to base questions on
    selected_docs = random.sample(documents, min(num_questions, len(documents)))
    
    for i, doc_data in enumerate(selected_docs, 1):
        doc_text = doc_data["text"][:800]
        metadata = doc_data["metadata"]
        code = metadata.get("code", "Code")
        article = metadata.get("article", "Article")
        
        prompt = f"""Tu es un expert en droit français. Basé sur cet extrait juridique, crée UNE question réaliste qu'un utilisateur pourrait poser.

Extrait juridique:
[{code} {article}]: {doc_text}

Règles pour la question:
1. Doit être answerable UNIQUEMENT avec les informations dans l'extrait ci-dessus
2. Doit tester la compréhension du contenu juridique
3. Doit être formulée comme une vraie question d'utilisateur (naturelle, pas trop technique)
4. Peut être une question de définition, de procédure, de conditions, ou de droits

Réponds UNIQUEMENT en JSON:
{{
  "question": "la question",
  "category": "definition|procedural|conditional|rights|temporal|general",
  "expected_aspects": ["aspect1", "aspect2"]
}}"""
        
        try:
            response = llm_chain.llm.invoke(prompt)
            content = getattr(response, "content", getattr(response, "text", str(response)))
            
            # Clean control characters from content
            content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')
            
            # Extract JSON
            json_start = content.find("{")
            json_end = content.rfind("}")
            if json_start >= 0 and json_end >= 0:
                json_str = content[json_start:json_end+1]
                # Remove any invalid escape sequences
                json_str = json_str.replace('\\', '\\\\').replace('\\\\"', '"').replace("\\'", "'")
                result = json.loads(json_str)
                
                question_data = {
                    "query": result.get("question", ""),
                    "category": result.get("category", "general"),
                    "expected_aspects": result.get("expected_aspects", []),
                    "source_code": code,
                    "source_article": article,
                    "in_context": True
                }
                
                if question_data["query"]:
                    questions.append(question_data)
                    print(f"  [{i}/{num_questions}] ✓ {question_data['query'][:60]}...")
                
        except Exception as e:
            print(f"  [{i}/{num_questions}] ✗ Error: {e}")
            continue
    
    return questions


def generate_out_of_context_questions(
    documents: List[Dict],
    llm_chain: LLMChainWrapper,
    num_questions: int = 3
) -> List[Dict]:
    """
    Generate questions that are NOT answerable from the indexed documents.
    These should be about legal topics that are likely NOT in the 5000 documents.
    """
    questions = []
    
    print(f"\nGenerating {num_questions} out-of-context questions...")
    
    # First, analyze what domains ARE in the documents
    codes_present = set()
    topics_present = []
    for doc in documents[:10]:
        code = doc["metadata"].get("code", "")
        if code:
            codes_present.add(code)
        # Get first 200 chars for topic analysis
        topics_present.append(doc["text"][:200])
    
    codes_str = ", ".join(codes_present) if codes_present else "Code du travail"
    
    # Generate out-of-context questions by asking about OTHER legal domains
    legal_domains_not_in_index = [
        "droit pénal",
        "droit fiscal", 
        "droit international",
        "droit constitutionnel",
        "droit de la propriété intellectuelle",
        "droit environnemental",
        "droit européen",
        "droit de la consommation",
        "droit des affaires complexes",
        "droit immobilier",
        "droit de la famille",
        "droit successorale"
    ]
    
    for i in range(num_questions):
        # Pick a domain NOT likely in the index
        domain = random.choice(legal_domains_not_in_index)
        
        prompt = f"""Tu es un expert en droit français. Crée UNE question réaliste sur le {domain}.

IMPORTANT: Cette question doit être sur UN SUJET QUI N'EST PAS dans le Code du travail ou les sujets de droit du travail courants.

Exemples de ce qui est dans l'index (à ÉVITER):
- {codes_str}
- contrats de travail, licenciement, période d'essai
- salaire, congés, durée du travail
- sécurité au travail, accidents

Crée une question sur {domain} qui serait légitime mais PAS couverte par les documents indexés.

Règles:
1. Question réaliste qu'un utilisateur pourrait poser
2. SUR UN SUJET DIFFÉRENT du droit du travail
3. Formulation naturelle

Réponds UNIQUEMENT en JSON:
{{
  "question": "la question",
  "category": "out_of_context",
  "expected_domain": "{domain}",
  "reason_why_out_of_context": "explication"
}}"""
        
        try:
            response = llm_chain.llm.invoke(prompt)
            content = getattr(response, "content", getattr(response, "text", str(response)))
            
            # Clean control characters from content
            content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')
            
            # Extract JSON
            json_start = content.find("{")
            json_end = content.rfind("}")
            if json_start >= 0 and json_end >= 0:
                json_str = content[json_start:json_end+1]
                # Remove any invalid escape sequences
                json_str = json_str.replace('\\', '\\\\').replace('\\\\"', '"').replace("\\'", "'")
                result = json.loads(json_str)
                
                question_data = {
                    "query": result.get("question", ""),
                    "category": "out_of_context",
                    "expected_domain": result.get("expected_domain", domain),
                    "reason": result.get("reason_why_out_of_context", ""),
                    "in_context": False
                }
                
                if question_data["query"]:
                    questions.append(question_data)
                    print(f"  [{i+1}/{num_questions}] ✓ {question_data['query'][:60]}...")
                    print(f"       Domain: {domain}")
                
        except Exception as e:
            print(f"  [{i+1}/{num_questions}] ✗ Error: {e}")
            # Create fallback question if LLM fails
            fallback_questions = [
                f"Quelles sont les règles de l'impôt sur le revenu en France?",
                f"Comment déposer un brevet d'invention?",
                f"Quels sont les droits du consommateur en cas de litige?",
                f"Comment fonctionne la succession en droit français?",
                f"Quelles sont les règles du divorce par consentement mutuel?",
            ]
            if i < len(fallback_questions):
                questions.append({
                    "query": fallback_questions[i],
                    "category": "out_of_context",
                    "expected_domain": domain,
                    "reason": f"Fallback question for {domain}",
                    "in_context": False
                })
                print(f"  [{i+1}/{num_questions}] ⚠ Using fallback question")
            continue
    
    return questions


def generate_edge_case_questions(
    llm_chain: LLMChainWrapper,
    num_questions: int = 2
) -> List[Dict]:
    """
    Generate edge case questions to test system robustness.
    These are intentionally tricky or malformed.
    """
    questions = []
    
    print(f"\nGenerating {num_questions} edge case questions...")
    
    edge_case_types = [
        "vague",  # Very vague query
        "ambiguous",  # Could mean multiple things
        "citation_only",  # Just a citation number
        "colloquial",  # Very informal language
    ]
    
    for i in range(num_questions):
        edge_type = random.choice(edge_case_types)
        
        prompts = {
            "vague": "Crée une question TRÈS VAGUE sur le droit du travail (1-2 mots seulement)",
            "ambiguous": "Crée une question AMBIGUË qui pourrait avoir plusieurs interprétations",
            "citation_only": "Crée une question qui est juste un numéro d'article (ex: 'Article L1234-1')",
            "colloquial": "Crée une question en langage TRÈS INFORMEL/courant (comme si un ami demandait)"
        }
        
        prompt = f"""{prompts[edge_type]}

Réponds UNIQUEMENT en JSON:
{{
  "question": "la question",
  "category": "edge_case",
  "edge_type": "{edge_type}"
}}"""
        
        try:
            response = llm_chain.llm.invoke(prompt)
            content = getattr(response, "content", getattr(response, "text", str(response)))
            
            # Clean control characters from content
            content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')
            
            json_start = content.find("{")
            json_end = content.rfind("}")
            if json_start >= 0 and json_end >= 0:
                json_str = content[json_start:json_end+1]
                # Remove any invalid escape sequences
                json_str = json_str.replace('\\', '\\\\').replace('\\\\"', '"').replace("\\'", "'")
                result = json.loads(json_str)
                
                question_data = {
                    "query": result.get("question", ""),
                    "category": "edge_case",
                    "edge_type": result.get("edge_type", edge_type),
                    "in_context": "unknown"  # Depends on the generated question
                }
                
                if question_data["query"]:
                    questions.append(question_data)
                    print(f"  [{i+1}/{num_questions}] ✓ ({edge_type}) {question_data['query'][:60]}...")
                
        except Exception as e:
            print(f"  [{i+1}/{num_questions}] ✗ Error: {e}")
            continue
    
    return questions


def save_queries(queries: List[Dict], output_path: str):
    """Save generated queries to JSON file"""
    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_queries": len(queries),
            "in_context_count": sum(1 for q in queries if q.get("in_context") == True),
            "out_of_context_count": sum(1 for q in queries if q.get("in_context") == False),
            "edge_case_count": sum(1 for q in queries if q.get("category") == "edge_case")
        },
        "queries": queries
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved {len(queries)} queries to: {output_path}")


def print_summary(queries: List[Dict]):
    """Print summary of generated queries"""
    print("\n" + "="*80)
    print("GENERATED TEST QUERIES SUMMARY")
    print("="*80)
    
    in_context = [q for q in queries if q.get("in_context") == True]
    out_context = [q for q in queries if q.get("in_context") == False]
    edge_cases = [q for q in queries if q.get("category") == "edge_case"]
    
    print(f"\n📊 Total: {len(queries)} queries")
    print(f"  ✓ In-context: {len(in_context)} (grounded in indexed documents)")
    print(f"  ✗ Out-of-context: {len(out_context)} (NOT in indexed documents)")
    print(f"  ⚠ Edge cases: {len(edge_cases)}")
    
    if in_context:
        print("\n\n📚 In-Context Questions:")
        print("-" * 80)
        for i, q in enumerate(in_context, 1):
            print(f"\n{i}. [{q.get('category', 'unknown')}] {q['query']}")
            print(f"   Source: {q.get('source_code', 'N/A')} {q.get('source_article', 'N/A')}")
            if q.get('expected_aspects'):
                print(f"   Aspects: {', '.join(q['expected_aspects'])}")
    
    if out_context:
        print("\n\n❌ Out-of-Context Questions:")
        print("-" * 80)
        for i, q in enumerate(out_context, 1):
            print(f"\n{i}. {q['query']}")
            print(f"   Expected domain: {q.get('expected_domain', 'N/A')}")
            if q.get('reason'):
                print(f"   Why out: {q['reason'][:80]}...")
    
    if edge_cases:
        print("\n\n⚠ Edge Case Questions:")
        print("-" * 80)
        for i, q in enumerate(edge_cases, 1):
            print(f"\n{i}. [{q.get('edge_type', 'unknown')}] {q['query']}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate test queries using indexed documents and Ollama"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="test_queries_generated.json",
        help="Output file for generated queries (default: test_queries_generated.json)"
    )
    parser.add_argument(
        "--num-in",
        type=int,
        default=5,
        help="Number of in-context questions to generate (default: 5)"
    )
    parser.add_argument(
        "--num-out",
        type=int,
        default=3,
        help="Number of out-of-context questions to generate (default: 3)"
    )
    parser.add_argument(
        "--num-edge",
        type=int,
        default=0,
        help="Number of edge case questions to generate (default: 0)"
    )
    parser.add_argument(
        "--sample-docs",
        type=int,
        default=20,
        help="Number of documents to sample from index (default: 20)"
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
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)"
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    print("="*80)
    print("TEST QUERY GENERATOR")
    print("="*80)
    print(f"Target: {args.num_in} in-context + {args.num_out} out-of-context + {args.num_edge} edge cases")
    
    # Initialize components
    print("\nInitializing components...")
    
    embedding_model = EmbeddingModel(
        model_name=args.embedding_model or config.EMBEDDING_MODEL_NAME
    )
    
    vector_store = FAISSVectorStore(
        embedding_dim=embedding_model.embedding_dim,
        index_path=args.index_path or config.FAISS_INDEX_PATH
    )
    
    llm_chain = LLMChainWrapper()
    
    print("✓ Components initialized")
    
    # Sample documents
    print("\n" + "="*80)
    sampled_docs = sample_documents_from_index(
        vector_store, embedding_model, args.sample_docs
    )
    
    if len(sampled_docs) < 5:
        print("ERROR: Not enough documents sampled. Is the index built?")
        print("Run: python main.py --build")
        return
    
    # Generate questions
    print("\n" + "="*80)
    print("GENERATING QUESTIONS")
    print("="*80)
    
    all_queries = []
    
    # Generate in-context questions
    if args.num_in > 0:
        in_context = generate_in_context_questions(
            sampled_docs, llm_chain, args.num_in
        )
        all_queries.extend(in_context)
    
    # Generate out-of-context questions
    if args.num_out > 0:
        out_context = generate_out_of_context_questions(
            sampled_docs, llm_chain, args.num_out
        )
        all_queries.extend(out_context)
    
    # Generate edge case questions
    if args.num_edge > 0:
        edge_cases = generate_edge_case_questions(llm_chain, args.num_edge)
        all_queries.extend(edge_cases)
    
    # Save and summarize
    print("\n" + "="*80)
    save_queries(all_queries, args.output)
    print_summary(all_queries)
    
    print("\n💡 Next steps:")
    print(f"   1. Review the generated questions in {args.output}")
    print(f"   2. Run comparison: python compare_retrieval_strategies.py --run-comparison --queries-file {args.output}")
    print(f"   3. Or test manually: python main.py --interactive")


if __name__ == "__main__":
    main()
