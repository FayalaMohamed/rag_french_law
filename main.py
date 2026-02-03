import os
import json
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
from chains.rag_chain import RAGPipeline, create_rag_pipeline
from utils.text_processing import chunk_documents
import config

try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    device = "cpu"
print(f"Using device: {device}")

def build_index(
    dataset_name: str = config.DATASET_NAME,
    output_index_path: str = config.FAISS_INDEX_PATH,
    processed_data_path: str = "data/processed_articles.json",
    limit: int = None,
    index_type: str = "hnsw_flat"
):
    """
    Build the RAG index from the French legal dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        output_index_path: Path to save the FAISS index
        processed_data_path: Path to save/load processed data
    """
    print("="*60)
    print("Building French Legal RAG Index")
    print("="*60)
    
    load_dotenv()
    
    print("\n1. Loading dataset...")
    dataset = load_french_legal_data(dataset_name)
    
    print("\n2. Filtering active articles...")
    filtered_dataset = filter_active_articles(dataset)
    
    print("\n3. Preprocessing articles...")
    if os.path.exists(processed_data_path):
        print(f"Loading processed data from {processed_data_path}")
        articles = load_processed_data(processed_data_path)
    else:
        articles = preprocess_articles(filtered_dataset)
        save_processed_data(articles, processed_data_path)
    
    if limit and len(articles) > limit:
        print(f"Limiting to {limit} articles for faster testing")
        articles = articles[:limit]
    
    print("\n4. Chunking articles...")
    print(f"   Chunk size: {config.CHUNK_SIZE} characters")
    print(f"   Chunk overlap: {config.CHUNK_OVERLAP} characters")
    chunked_articles = chunk_documents(
        articles,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        text_field="text",
        metadata_fields=["code", "article"]
    )
    print(f"   Created {len(chunked_articles)} chunks from {len(articles)} articles")
    
    print("\n5. Initializing embedding model...")
    embedding_model = EmbeddingModel(device=device)
    
    print("\n5. Creating vector store...")
    print(f"   Using index type: {index_type}")
    vector_store = FAISSVectorStore(
        embedding_dim=embedding_model.embedding_dim,
        index_path=output_index_path,
        index_type=index_type
    )
    
    print("\n6. Embedding and indexing documents...")
    texts = [chunk["text"] for chunk in chunked_articles]
    metadatas = [chunk["metadata"] for chunk in chunked_articles]
    
    embeddings = embedding_model.embed_documents(texts)
    vector_store.add_documents(texts, embeddings, metadatas)
    
    print("\n7. Saving index...")
    vector_store.save_index(output_index_path)
    
    print("\n" + "="*60)
    print(f"Index built successfully with {vector_store.index.ntotal} documents")
    print(f"Index saved to: {output_index_path}")
    print("="*60)
    
    return vector_store


def load_pipeline(
    index_path: str = config.FAISS_INDEX_PATH,
    embedding_model_name: str = config.EMBEDDING_MODEL_NAME
) -> RAGPipeline:
    load_dotenv()
    
    return create_rag_pipeline(
        faiss_index_path=index_path,
        embedding_model_name=embedding_model_name
    )


def interactive_query(pipeline: RAGPipeline):
    print("\n" + "="*60)
    print("French Legal RAG - Interactive Query Mode")
    print("Type 'quit' to exit")
    print("="*60)
    
    while True:
        query = input("\nEnter your legal question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        print("\nProcessing query...")
        result = pipeline.answer(query, k=config.TOP_K_RETRIEVAL)
        
        print("\n" + "-"*60)
        print("ANSWER:")
        print(result["answer"])
        print("\nSOURCES:")
        for i, source in enumerate(result["sources"], 1):
            print(f"{i}. [{source['code']} {source['article']}] "
                  f"(distance: {source['distance']:.4f})")
        print("-"*60)


def test_pipeline(pipeline: RAGPipeline):
    test_questions = [
        "Quelles sont les conditions de la période d'essai en droit du travail?",
        "Comment rompre un contrat de travail à durée déterminée?",
        "Quelles sont les obligations de l'employeur en matière de sécurité?",
        "Qu'est-ce que le harcèlement moral au travail?"
    ]
    
    print("\n" + "="*60)
    print("Running Pipeline Tests")
    print("="*60)
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-"*40)
        
        result = pipeline.answer(question, k=config.TOP_K_RETRIEVAL)
        
        print(f"Answer: {result['answer'][:300]}...")
        print(f"\n Sources retrieved: {result['num_sources_retrieved']}")
        
        i = 1
        for source in result['sources']:
            print(f"Source {i} : {source['text']}")
            i += 1
            
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="French Legal RAG Pipeline")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the RAG index from dataset"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of articles for faster testing"
    )
    parser.add_argument(
        "--index_type",
        type=str,
        default="hnsw_flat",
        choices=["flat_l2", "ivf_flat", "ivf_pq", "hnsw_flat", "lsh"],
        help="FAISS index type to use (default: hnsw_flat)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test queries"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive query mode"
    )
    
    args = parser.parse_args()
    
    if args.build:
        build_index(limit=args.limit, index_type=args.index_type)
    elif args.test:
        pipeline = load_pipeline()
        test_pipeline(pipeline)
    elif args.interactive:
        pipeline = load_pipeline()
        interactive_query(pipeline)
    else:
        print("French Legal RAG Pipeline")
        print("\nUsage:")
        print("  python main.py --build [--index_type TYPE]  # Build the index")
        print("  python main.py --test                       # Run test queries")
        print("  python main.py --interactive                # Interactive query mode")
        print("\nIndex types: hnsw_flat (default), flat_l2, ivf_flat, ivf_pq, lsh")
        print("\nExamples:")
        print("  python main.py --build --index_type hnsw_flat --limit 1000")
        print("  python main.py --build --index_type ivf_flat")
