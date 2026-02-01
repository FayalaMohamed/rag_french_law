from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import os
import pickle
import torch
import config


def get_device() -> str:
    """Get the appropriate device for PyTorch models."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class EmbeddingModel:
    """
    Wrapper for sentence embedding model optimized for French text
    """
    
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME, device: Union[str, None] = None):
        self.model_name = model_name
        if device is None:
            device = get_device()
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Loaded embedding model: {model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Device: {self.model.device}")
    
    def embed_documents(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=batch_size
        )
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings
    
    def embed_query(self, text: str) -> np.ndarray:
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True
        )
        return embedding[0]
    
    def embed_hypothetical_document(self, question: str) -> np.ndarray:
        """
        Generate hypothetical document for HyDE (Hypothetical Document Embeddings)
        
        Args:
            question: User question
            
        Returns:
            Hypothetical document embedding
        """
        hypothetical_doc = f"Answer to legal question: {question}"
        return self.embed_query(hypothetical_doc)
    
    def save_embeddings(self, embeddings: np.ndarray, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        np.save(path, embeddings)
        print(f"Saved embeddings to {path}")
    
    def load_embeddings(self, path: str) -> np.ndarray:
        return np.load(path)
    
    def save_model(self, output_path: str):
        self.model.save(output_path)
        print(f"Model saved to {output_path}")


if __name__ == "__main__":
    model = EmbeddingModel()
    test_embedding = model.embed_query("Comment rédiger un contrat de travail?")
    print(f"Test embedding shape: {test_embedding.shape}")
    print(f"Test embedding: {test_embedding}")
