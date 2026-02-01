import os
from typing import Optional, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import config


class LLMChainWrapper:
    """
    Ollama-based wrapper for local LLM.
    Provides HyDE generation, question decomposition, and final answer generation.
    """

    def __init__(
        self,
        model_name: str = config.LLM_MODEL,
        temperature: float = config.LLM_TEMPERATURE,
        base_url: str = config.OLLAMA_BASE_URL,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = base_url

        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
        )

        from prompts.prompts import (
            get_hyde_prompt,
            get_question_decomposition_prompt,
        )
        self.hyde_prompt = get_hyde_prompt()
        self.decomposition_prompt = get_question_decomposition_prompt()

    def generate_hypothetical_answer(self, question: str) -> str:
        """Generate a hypothetical HyDE answer."""
        messages = self.hyde_prompt.format_prompt(question=question).to_messages()
        response = self.llm.invoke(messages)
        return getattr(response, "content", getattr(response, "text", ""))

    def decompose_question(self, question: str) -> list:
        """Decompose a question into sub-questions."""
        messages = self.decomposition_prompt.format_prompt(question=question).to_messages()
        response = self.llm.invoke(messages)
        text = getattr(response, "content", getattr(response, "text", ""))
        sub_questions = [q.strip() for q in text.split(';') if q.strip()]
        return sub_questions

    def generate_answer(
        self,
        question: str,
        context: str,
        prompt: Optional[Any] = None,
    ) -> str:
        system_prompt = """Vous êtes un assistant juridique français expert. 
Votre rôle est de répondre aux questions légales en utilisant UNIQUEMENT les informations 
contenues dans les documents juridiques fournis.

Instructions importantes:
1. Basez votre réponse UNIQUEMENT sur les documents juridiques fournis
2. Citez les articles de loi pertinents avec leurs références exactes
3. Si l'information n'est pas dans les documents, indiquez-le clairement
4. Structurez votre réponse de manière claire et professionnelle
5. Mentionnez la source juridique (Code, article) pour chaque affirmation

Documents contextuels:
{context}

Question: {question}

Réponse juridique:"""

        system_content = system_prompt.format(context=context, question=question)
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=question),
        ]
        response = self.llm.invoke(messages)
        return getattr(response, "content", getattr(response, "text", ""))


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    chain = LLMChainWrapper()

    hypothetical = chain.generate_hypothetical_answer(
        "Quelles sont les conditions de la période d'essai?"
    )
    print("Hypothetical answer:")
    print(hypothetical)

    sub_questions = chain.decompose_question(
        "Quelles sont les règles concernant le contrat de travail à durée déterminée?"
    )
    print("\nDecomposed questions:")
    for q in sub_questions:
        print(f"- {q}")
