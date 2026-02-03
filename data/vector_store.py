import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional, Literal
import os
import json
from datetime import datetime
import config


class FAISSVectorStore:
    """
    FAISS-based vector store for document retrieval
    Supports metadata filtering and efficient similarity search
    Supports multiple FAISS index types for performance testing
    """
    
    SUPPORTED_INDEX_TYPES = [
        "flat_l2",           # Exact search - baseline, most accurate
        "ivf_flat",          # Inverted file index with flat encoding
        "ivf_pq",            # Inverted file index with product quantization
        "hnsw_flat",         # Hierarchical Navigable Small World graph
        "lsh",               # Locality Sensitive Hashing
    ]
    
    def __init__(
        self, 
        embedding_dim: int, 
        index_path: str = None,
        index_type: Literal["flat_l2", "ivf_flat", "ivf_pq", "hnsw_flat", "lsh"] = "hnsw_flat",
        nlist: int = 100,           # For IVF indices: number of clusters
        nprobe: int = 10,           # For IVF indices: clusters to search
        nbits: int = 8,             # For LSH: number of hash bits
        ef_construction: int = 40,  # For HNSW: construction-time quality
        ef_search: int = 16,        # For HNSW: search-time quality
        m: int = 16,                # For HNSW: connections per layer
        pq_m: int = 8,              # For IVF_PQ: number of subquantizers
        pq_nbits: int = 8,          # For IVF_PQ: bits per subquantizer
    ):
        """
        Initialize FAISS vector store with configurable index type.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_path: Path to save/load index
            index_type: Type of FAISS index (flat_l2, ivf_flat, ivf_pq, hnsw_flat, lsh)
            nlist: Number of clusters for IVF indices
            nprobe: Number of clusters to search for IVF indices
            nbits: Number of hash bits for LSH
            ef_construction: Construction-time quality for HNSW
            ef_search: Search-time quality for HNSW
            m: Number of connections per layer for HNSW
            pq_m: Number of subquantizers for IVF_PQ
            pq_nbits: Bits per subquantizer for IVF_PQ
        """
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.index_type = index_type
        self.documents = []
        self.metadatas = []
        
        self.nlist = nlist
        self.nprobe = nprobe
        self.nbits = nbits
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m = m
        self.pq_m = pq_m
        self.pq_nbits = pq_nbits
        
        self.index = None
        self.is_trained = False
        
        if index_path and os.path.exists(index_path):
            self.load_index()
        else:
            self._create_index()
    
    def _create_index(self):
        if self.index_type == "flat_l2":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.is_trained = True
            
        elif self.index_type == "ivf_flat":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist)
            self.is_trained = False
            
        elif self.index_type == "ivf_pq":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFPQ(
                quantizer, 
                self.embedding_dim, 
                self.nlist, 
                self.pq_m, 
                self.pq_nbits
            )
            self.is_trained = False
            
        elif self.index_type == "hnsw_flat":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, self.m)
            self.index.hnsw.efConstruction = self.ef_construction
            self.index.hnsw.efSearch = self.ef_search
            self.is_trained = True
            
        elif self.index_type == "lsh":
            self.index = faiss.IndexLSH(self.embedding_dim, self.nbits)
            self.is_trained = True
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}. "
                           f"Supported types: {self.SUPPORTED_INDEX_TYPES}")
        
        print(f"Created {self.index_type} index with dimension {self.embedding_dim}")
    
    def _train_index(self, embeddings: np.ndarray):
        """Train the index if required (for IVF indices)."""
        if not self.is_trained and hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if embeddings.shape[0] < self.nlist:
                print(f"Warning: Not enough samples ({embeddings.shape[0]}) for nlist={self.nlist}. "
                      f"Reducing nlist to {max(1, embeddings.shape[0] // 2)}")
                # Recreate with smaller nlist
                self.nlist = max(1, embeddings.shape[0] // 2)
                self._create_index()
            
            print(f"Training {self.index_type} index with {embeddings.shape[0]} vectors...")
            self.index.train(embeddings)
            self.is_trained = True
            print("Training complete")
    
    def _set_search_params(self):
        if self.index_type in ["ivf_flat", "ivf_pq"]:
            self.index.nprobe = min(self.nprobe, self.nlist)
        elif self.index_type == "hnsw_flat":
            self.index.hnsw.efSearch = self.ef_search
    
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
        
        self._train_index(embeddings)
        
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
        
        self._set_search_params()
        
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
        
        # Save index metadata
        metadata = {
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "nbits": self.nbits,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "m": self.m,
            "pq_m": self.pq_m,
            "pq_nbits": self.pq_nbits,
            "is_trained": self.is_trained
        }
        with open(os.path.join(save_path, "index_metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
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
        
        # Load index metadata if available
        metadata_path = os.path.join(load_path, "index_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.index_type = metadata.get("index_type", "flat_l2")
                self.embedding_dim = metadata.get("embedding_dim", self.embedding_dim)
                self.nlist = metadata.get("nlist", 100)
                self.nprobe = metadata.get("nprobe", 10)
                self.nbits = metadata.get("nbits", 8)
                self.ef_construction = metadata.get("ef_construction", 40)
                self.ef_search = metadata.get("ef_search", 16)
                self.m = metadata.get("m", 16)
                self.pq_m = metadata.get("pq_m", 8)
                self.pq_nbits = metadata.get("pq_nbits", 8)
                self.is_trained = metadata.get("is_trained", True)
        
        print(f"Loaded {self.index_type} index with {self.index.ntotal} documents from {load_path}")
    
    def get_stats(self) -> Dict:
        stats = {
            "total_documents": self.index.ntotal if self.index else 0,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "index_path": self.index_path,
            "is_trained": self.is_trained
        }
        
        if self.index_type in ["ivf_flat", "ivf_pq"]:
            stats["nlist"] = self.nlist
            stats["nprobe"] = self.nprobe
        elif self.index_type == "hnsw_flat":
            stats["ef_construction"] = self.ef_construction
            stats["ef_search"] = self.ef_search
            stats["m"] = self.m
        elif self.index_type == "lsh":
            stats["nbits"] = self.nbits
        
        if self.index_type == "ivf_pq":
            stats["pq_m"] = self.pq_m
            stats["pq_nbits"] = self.pq_nbits
        
        return stats
    
    def clear(self):
        self._create_index()
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
        k=3
    )
    
    print("\nSearch results:")
    for doc, dist in zip(results, distances):
        print(f"- Distance: {dist:.4f}")
        print(f"  Document: {doc[:100]}...")
