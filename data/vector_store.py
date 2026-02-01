import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import json
from datetime import datetime
import config


class FAISSVectorStore:
    """
    FAISS-based vector store for document retrieval
    Supports metadata filtering and efficient similarity search
    """
    
    def __init__(self, embedding_dim: int, index_path: str = None):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.index = None
        self.documents = []
        self.metadatas = []
        
        if index_path and os.path.exists(index_path):
            self.load_index()
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict]] = None
    ):
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"Number of documents ({len(documents)}) "
                f"doesn't match embeddings ({embeddings.shape[0]})"
            )
        
        self.index.add(embeddings)
        self.documents.extend(documents)
        
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{}] * len(documents))
        
        print(f"Added {len(documents)} documents. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = config.TOP_K_RETRIEVAL
    ) -> Tuple[List[str], List[float], List[Dict]]:
        k = min(k, self.index.ntotal)
        
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            k
        )
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "distance": float(dist),
                    "metadata": self.metadatas[idx]
                })
        
        documents = [r["document"] for r in results]
        distances = [r["distance"] for r in results]
        metadatas = [r["metadata"] for r in results]
        
        return documents, distances, metadatas
    
    def search_by_text(
        self,
        query_text: str,
        embedding_model,
        k: int = config.TOP_K_RETRIEVAL
    ) -> Tuple[List[str], List[float], List[Dict]]:
        query_embedding = embedding_model.embed_query(query_text)
        return self.search(query_embedding, k)
    
    def search_with_hyde(
        self,
        question: str,
        embedding_model,
        llm_chain,
        k: int = config.TOP_K_RETRIEVAL
    ) -> Tuple[List[str], List[float], List[Dict]]:
        hypothetical_doc = llm_chain.generate_hypothetical_answer(question)
        query_embedding = embedding_model.embed_query(hypothetical_doc)
        return self.search(query_embedding, k)
    
    def save_index(self, path: str = None):
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No save path specified")
        
        os.makedirs(save_path, exist_ok=True)
        
        faiss.write_index(self.index, os.path.join(save_path, "index.faiss"))
        
        with open(os.path.join(save_path, "documents.json"), 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False)
        
        with open(os.path.join(save_path, "metadatas.json"), 'w', encoding='utf-8') as f:
            json.dump(self.metadatas, f, ensure_ascii=False)
        
        print(f"Saved index to {save_path}")
    
    def load_index(self, path: str = None):
        load_path = path or self.index_path
        if not load_path or not os.path.exists(load_path):
            print(f"No existing index found at {load_path}")
            return
        
        self.index = faiss.read_index(os.path.join(load_path, "index.faiss"))
        
        with open(os.path.join(load_path, "documents.json"), 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        with open(os.path.join(load_path, "metadatas.json"), 'r', encoding='utf-8') as f:
            self.metadatas = json.load(f)
        
        print(f"Loaded index with {self.index.ntotal} documents from {load_path}")
    
    def get_stats(self) -> Dict:
        return {
            "total_documents": self.index.ntotal if self.index else 0,
            "embedding_dim": self.embedding_dim,
            "index_path": self.index_path
        }
    
    def clear(self):
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.documents = []
        self.metadatas = []


if __name__ == "__main__":
    from models.embeddings import EmbeddingModel
    
    embedding_model = EmbeddingModel()
    vector_store = FAISSVectorStore(
        embedding_dim=embedding_model.embedding_dim,
        index_path="data/faiss_index"
    )
    
    test_docs = [
        "Article L1234-1 du Code du travail: La période d'essai permet aux parties d'évaluer les conditions de travail.",
        "Article 1101 du Code civil: Le contrat est une convention par laquelle une ou plusieurs personnes s'engagent.",
        "Article 1240 du Code civil: Tout fait quelconque de l'homme qui cause à autrui un dommage."
    ]
    
    embeddings = embedding_model.embed_documents(test_docs)
    vector_store.add_documents(test_docs, embeddings)
    
    results, distances, metadatas = vector_store.search_by_text(
        "Qu'est-ce que la période d'essai?",
        embedding_model,
        k=2
    )
    
    print("\nSearch results:")
    for doc, dist in zip(results, distances):
        print(f"- Distance: {dist:.4f}")
        print(f"  Document: {doc[:100]}...")
