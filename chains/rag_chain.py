from typing import Dict, List, Tuple, Optional
import json
from data.vector_store import FAISSVectorStore
from models.embeddings import EmbeddingModel
from chains.llm_chain import LLMChainWrapper
import config


class RAGPipeline:
    """
    Main RAG pipeline for French legal document question answering.
    Combines retrieval, question decomposition, and generation.
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: FAISSVectorStore,
        llm_chain: LLMChainWrapper
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: Embedding model instance
            vector_store: FAISS vector store instance
            llm_chain: LLM chain wrapper instance
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_chain = llm_chain
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        use_hyde: bool = False
    ) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            use_hyde: Whether to use HyDE for retrieval
            
        Returns:
            Tuple of (documents, distances, metadatas)
        """
        if use_hyde:
            return self.vector_store.search_with_hyde(
                query,
                self.embedding_model,
                self.llm_chain,
                k=k
            )
        else:
            return self.vector_store.search_by_text(
                query,
                self.embedding_model,
                k=k
            )
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose a query into sub-queries for better retrieval.
        
        Args:
            query: User query
            
        Returns:
            List of sub-queries
        """
        return self.llm_chain.decompose_question(query)
    
    def retrieve_with_decomposition(
        self,
        query: str,
        k: int = 5
    ) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Retrieve documents using query decomposition.
        
        Args:
            query: User query
            k: Documents per sub-query
            
        Returns:
            Combined retrieval results
        """
        sub_queries = self.decompose_query(query)
        all_results = []
        all_distances = []
        all_metadatas = []
        
        for sub_query in sub_queries:
            docs, dists, metas = self.retrieve(sub_query, k=k)
            all_results.extend(docs)
            all_distances.extend(dists)
            all_metadatas.extend(metas)
        
        unique_results = []
        seen_texts = set()
        for doc, dist, meta in zip(all_results, all_distances, all_metadatas):
            if doc not in seen_texts:
                seen_texts.add(doc)
                unique_results.append((doc, dist, meta))
        
        unique_results.sort(key=lambda x: x[1])
        
        final_docs = [r[0] for r in unique_results[:k]]
        final_dists = [r[1] for r in unique_results[:k]]
        final_metas = [r[2] for r in unique_results[:k]]
        
        return final_docs, final_dists, final_metas
    
    def generate(
        self,
        query: str,
        context: str,
        use_refinement: bool = False
    ) -> str:
        """
        Generate an answer using the retrieved context.
        
        Args:
            query: User query
            context: Combined retrieved documents
            use_refinement: Whether to use answer refinement
            
        Returns:
            Generated answer string
        """
        return self.llm_chain.generate_answer(query, context)
    
    def answer(
        self,
        query: str,
        k: int = 5,
        use_hyde: bool = True,
        use_decomposition: bool = True
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve and generate answer.
        
        Args:
            query: User question
            k: Number of documents to retrieve
            use_hyde: Whether to use HyDE
            use_decomposition: Whether to use query decomposition
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if use_decomposition:
            retrieved_docs, distances, metadatas = self.retrieve_with_decomposition(
                query, k=k
            )
        else:
            retrieved_docs, distances, metadatas = self.retrieve(
                query, k=k, use_hyde=use_hyde
            )
        
        context = "\n\n".join(retrieved_docs)
        
        answer = self.generate(query, context)
        
        sources = []
        for doc, meta in zip(retrieved_docs, metadatas):
            source = {
                "text": doc[:200] + "..." if len(doc) > 200 else doc,
                "code": meta.get("code", "Unknown"),
                "article": meta.get("article", "Unknown"),
                "distance": float(distances[retrieved_docs.index(doc)])
            }
            sources.append(source)
        
        return {
            "answer": answer,
            "sources": sources,
            "num_sources_retrieved": len(retrieved_docs)
        }
    
    def build_index_from_dataset(
        self,
        dataset_path: str,
        text_field: str = "text",
        metadata_fields: List[str] = None
    ):
        """
        Build the FAISS index from a dataset.
        
        Args:
            dataset_path: Path to dataset or processed JSON file
            text_field: Field name for text content
            metadata_fields: Fields to include in metadata
        """
        import json
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item[text_field] for item in data]
        
        metadatas = []
        for item in data:
            meta = {}
            if metadata_fields:
                for field in metadata_fields:
                    if field in item:
                        meta[field] = item[field]
            metadatas.append(meta)
        
        print(f"Embedding {len(texts)} documents...")
        embeddings = self.embedding_model.embed_documents(texts)
        
        print("Adding documents to vector store...")
        self.vector_store.add_documents(texts, embeddings, metadatas)
        
        print(f"Index built with {self.vector_store.index.ntotal} documents")
    
    def save_index(self, path: str = None):
        """Save the vector store index."""
        self.vector_store.save_index(path)
    
    def load_index(self, path: str = None):
        """Load the vector store index."""
        self.vector_store.load_index(path)


def create_rag_pipeline(
    faiss_index_path: str = config.FAISS_INDEX_PATH,
    embedding_model_name: str = config.EMBEDDING_MODEL_NAME
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline.
    
    Args:
        faiss_index_path: Path to FAISS index
        embedding_model_name: Name of embedding model
        
    Returns:
        Configured RAGPipeline instance
    """
    embedding_model = EmbeddingModel(embedding_model_name)
    
    vector_store = FAISSVectorStore(
        embedding_dim=embedding_model.embedding_dim,
        index_path=faiss_index_path
    )
    
    llm_chain = LLMChainWrapper()
    
    return RAGPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm_chain=llm_chain
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    pipeline = create_rag_pipeline(
        faiss_index_path="data/faiss_index"
    )
    
    result = pipeline.answer(
        "Quelles sont les conditions de rupture du contrat de travail à durée déterminée?",
        k=3
    )
    
    print("\n" + "="*50)
    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES:")
    for source in result["sources"]:
        print(f"- {source['code']} {source['article']} (distance: {source['distance']:.4f})")
