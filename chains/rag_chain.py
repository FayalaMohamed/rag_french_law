from typing import Dict, List, Tuple
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
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_chain = llm_chain
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        use_hyde: bool = False
    ) -> Tuple[List[str], List[float], List[Dict]]:
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
        return self.llm_chain.decompose_question(query)
    
    def retrieve_with_decomposition(
        self,
        query: str,
        k: int = 5
    ) -> Tuple[List[str], List[float], List[Dict]]:
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
        return self.llm_chain.generate_answer(query, context)
    
    def answer(
        self,
        query: str,
        k: int = 5,
        use_hyde: bool = True,
        use_decomposition: bool = True
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve and generate answer
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

def create_rag_pipeline(
    faiss_index_path: str = config.FAISS_INDEX_PATH,
    embedding_model_name: str = config.EMBEDDING_MODEL_NAME
) -> RAGPipeline:
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
