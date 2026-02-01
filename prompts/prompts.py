from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


QUESTION_DECOMPOSITION_PROMPT = PromptTemplate(
    template="""Vous êtes un expert en droit français. Décomposez la question suivante en sous-questions plus précises 
qui aideront à trouver les réponses légales pertinentes.

Question: {question}

Sous-questions décomposées (séparées par des points-virgules):
""",
    input_variables=["question"]
)


HYDE_GENERATION_PROMPT = PromptTemplate(
    template="""Générez une réponse hypothétique détaillée à la question légale suivante.
Cette réponse sera utilisée pour améliorer la recherche de documents.

Question: {question}

Réponse hypothétique détaillée:
""",
    input_variables=["question"]
)


RAG_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Vous êtes un assistant juridique français expert. 
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

Réponse juridique:
"""),
    ("human", "{question}")
])


REFINEMENT_PROMPT = PromptTemplate(
    template="""Vous améliorez les réponses juridiques en les rendant plus complètes et précises.

Réponse initiale:
{initial_answer}

Contexte supplémentaire trouvé:
{additional_context}

Réponse améliorée et finalisée:
""",
    input_variables=["initial_answer", "additional_context"]
)


def get_rag_prompt() -> ChatPromptTemplate:
    """Get the main RAG prompt template."""
    return RAG_ANSWER_PROMPT


def get_hyde_prompt() -> PromptTemplate:
    """Get the HyDE hypothetical document generation prompt."""
    return HYDE_GENERATION_PROMPT


def get_question_decomposition_prompt() -> PromptTemplate:
    """Get the question decomposition prompt."""
    return QUESTION_DECOMPOSITION_PROMPT


if __name__ == "__main__":
    print("RAG Prompt Template:")
    print(RAG_ANSWER_PROMPT.format(
        context="Article L1234-1 du Code du travail...",
        question="Qu'est-ce que la période d'essai?"
    ))
