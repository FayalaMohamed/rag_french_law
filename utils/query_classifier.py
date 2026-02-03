"""
Query classification and routing for legal RAG.
Provides functionality to classify queries and route them to appropriate handlers.
"""

from typing import List, Dict, Tuple, Optional
from enum import Enum
import re


class QueryType(Enum):
    """Types of legal queries."""
    DEFINITION = "definition"
    PROCEDURAL = "procedural"
    CONDITIONAL = "conditional"
    RIGHTS_OBLIGATIONS = "rights_obligations"
    TEMPORAL = "temporal"
    COMPARATIVE = "comparative"
    HISTORICAL = "historical"
    CITATION_LOOKUP = "citation_lookup"
    GENERAL = "general"


class LegalDomain(Enum):
    """Legal domains/areas of French law."""
    LABOR_LAW = "labor_law"
    CIVIL_LAW = "civil_law"
    COMMERCIAL_LAW = "commercial_law"
    PENAL_LAW = "penal_law"
    ADMINISTRATIVE_LAW = "administrative_law"
    CONSTITUTIONAL_LAW = "constitutional_law"
    TAX_LAW = "tax_law"
    SOCIAL_SECURITY = "social_security"
    HOUSING_LAW = "housing_law"
    FAMILY_LAW = "family_law"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    ENVIRONMENTAL_LAW = "environmental_law"
    EUROPEAN_LAW = "european_law"
    INTERNATIONAL_LAW = "international_law"
    GENERAL = "general"


class QueryClassifier:
    """Classifies legal queries by type and domain."""

    DEFINITION_PATTERNS = [
        r"qu'est[- ]ce que",
        r"c'est quoi",
        r"définition de",
        r"signifie",
        r"veut dire",
        r"que signifie",
        r"explain",
        r"what is",
    ]

    PROCEDURAL_PATTERNS = [
        r"comment (faire|procéder|obtenir)",
        r"quelle est la procédure",
        r"comment (rompre|résilier|contacter)",
        r"comment (rédiger|préparer)",
        r"steps? to",
        r"how to",
        r"procédure",
        r"formalités",
    ]

    CONDITIONAL_PATTERNS = [
        r"condition",
        r"exiger",
        r"si\s+\w+",
        r"lorsque",
        r"à condition",
        r"à défaut",
        r"peut[- ]on",
        r"est[- ]il possible",
        r"quand peut",
        r"qui peut",
    ]

    RIGHTS_OBLIGATIONS_PATTERNS = [
        r"droit",
        r"obligat",
        r"devoir",
        r"responsabilité",
        r"peut\s+\w+",
        r"ne peut pas",
        r"interdit",
        r"autorisé",
    ]

    TEMPORAL_PATTERNS = [
        r"quand",
        r"délai",
        r"durée",
        r"délais",
        r"date (limite|de)",
        r"combien de temps",
        r"prescription",
        r"时效",
        r"temporal",
        r"within\s+\d+\s+(day|week|month|year)",
    ]

    COMPARATIVE_PATTERNS = [
        r"différence (entre|entre les)",
        r"compar",
        r"versus",
        r"vs\.?",
        r"ou bien",
        r"alternative",
        r"choice between",
    ]

    HISTORICAL_PATTERNS = [
        r"historique",
        r"évolution",
        r"origines?",
        r"ancien",
        r"loi\s+(n°)?\s*\d{4}",
        r"réforme",
        r"modification",
    ]

    CITATION_PATTERNS = [
        r"article\s+\w+",
        r"art\.?\s*\w+",
        r"code\s+\w+",
        r"loi\s+n°",
        r"décret\s+n°",
        r"jurisprudence",
        r"citation",
    ]

    LABOR_LAW_KEYWORDS = [
        "travail", "employeur", "employé", "salarié", "contrat de travail",
        "licenciement", "rupture", "période d'essai", "salaire", "congé",
        "représentant du personnel", "syndicat", "convention collective",
        "URE", "prud'hommes", "chômage", "retraite", "formation professionnelle",
        "CDD", "CDI", "intérim", "télétravail", "accident du travail"
    ]

    CIVIL_LAW_KEYWORDS = [
        "civil", "contrat", "obligations", "propriété", "succession",
        "famille", "mariage", "divorce", "adotpion", "tutelle",
        "responsabilité", "dommages-intérêts", "prêt", "bail", "vente"
    ]

    COMMERCIAL_LAW_KEYWORDS = [
        "commercial", "société", "commerce", "entreprise", "fonds de commerce",
        "cession", "fusaion", "faillite", "redressement", "liquidation",
        "mandataire", "insolvabilité", "TPE", "PME", "statuts"
    ]

    PENAL_LAW_KEYWORDS = [
        "pénal", "crime", "délit", "contravention", "peine", "prison",
        "amende", "sanction", "condamnation", "acquittement", "instruction",
        "enquête", "police judiciaire", "parquet", "procureur"
    ]

    ADMINISTRATIVE_LAW_KEYWORDS = [
        "administratif", "administration", "acte administratif", "recours",
        "juridiction administrative", "tribunal administratif", "CE",
        "décision administrative", "autorisation", "permis", "urbanisme"
    ]

    def __init__(self):
        """Initialize the query classifier."""
        self.query_patterns = {
            QueryType.DEFINITION: self.DEFINITION_PATTERNS,
            QueryType.PROCEDURAL: self.PROCEDURAL_PATTERNS,
            QueryType.CONDITIONAL: self.CONDITIONAL_PATTERNS,
            QueryType.RIGHTS_OBLIGATIONS: self.RIGHTS_OBLIGATIONS_PATTERNS,
            QueryType.TEMPORAL: self.TEMPORAL_PATTERNS,
            QueryType.COMPARATIVE: self.COMPARATIVE_PATTERNS,
            QueryType.HISTORICAL: self.HISTORICAL_PATTERNS,
            QueryType.CITATION_LOOKUP: self.CITATION_PATTERNS,
        }

        self.domain_keywords = {
            LegalDomain.LABOR_LAW: self.LABOR_LAW_KEYWORDS,
            LegalDomain.CIVIL_LAW: self.CIVIL_LAW_KEYWORDS,
            LegalDomain.COMMERCIAL_LAW: self.COMMERCIAL_LAW_KEYWORDS,
            LegalDomain.PENAL_LAW: self.PENAL_LAW_KEYWORDS,
            LegalDomain.ADMINISTRATIVE_LAW: self.ADMINISTRATIVE_LAW_KEYWORDS,
        }

    def classify_query_type(self, query: str) -> Tuple[QueryType, float]:
        """
        Classify the type of a query.

        Args:
            query: User query

        Returns:
            Tuple of (QueryType, confidence_score)
        """
        query_lower = query.lower()
        scores = {}

        for qtype, patterns in self.query_patterns.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1.0
            scores[qtype] = score

        if max(scores.values()) == 0:
            return QueryType.GENERAL, 0.5

        best_type = max(scores, key=scores.get)
        confidence = min(1.0, scores[best_type] * 0.5 + 0.3)

        return best_type, confidence

    def classify_domain(self, query: str) -> Tuple[LegalDomain, float]:
        """
        Classify the legal domain of a query.

        Args:
            query: User query

        Returns:
            Tuple of (LegalDomain, confidence_score)
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1.0
                if keyword in query_words:
                    score += 0.5
            scores[domain] = score

        if max(scores.values()) == 0:
            return LegalDomain.GENERAL, 0.3

        best_domain = max(scores, key=scores.get)
        confidence = min(0.95, scores[best_domain] * 0.15 + 0.4)

        return best_domain, confidence

    def classify(self, query: str) -> Dict:
        """
        Complete query classification.

        Args:
            query: User query

        Returns:
            Dictionary with classification results
        """
        query_type, type_confidence = self.classify_query_type(query)
        domain, domain_confidence = self.classify_domain(query)

        urgency_keywords = ["urgent", "immédiat", "délai", "urgent", "asap"]
        has_urgency = any(kw in query.lower() for kw in urgency_keywords)

        complexity_keywords = [
            "complexe", "compliqué", "plusieurs", "multiples",
            "différentes", "divers", "plusieurs aspects"
        ]
        complexity = "complex" if any(kw in query.lower() for kw in complexity_keywords) else "simple"

        return {
            "query_type": query_type.value,
            "query_type_confidence": type_confidence,
            "domain": domain.value,
            "domain_confidence": domain_confidence,
            "is_urgent": has_urgency,
            "complexity": complexity,
            "original_query": query,
        }


class QueryRouter:
    """Routes queries to appropriate retrieval strategies."""

    def __init__(self):
        """Initialize the query router."""
        self.classifier = QueryClassifier()
        self.retrieval_strategies = {}

    def register_strategy(self, query_type: QueryType, strategy_name: str):
        """
        Register a retrieval strategy for a query type.

        Args:
            query_type: Query type to handle
            strategy_name: Name of the strategy
        """
        self.retrieval_strategies[query_type] = strategy_name

    def get_retrieval_params(self, classification: Dict) -> Dict:
        """
        Get retrieval parameters based on query classification.

        Args:
            classification: Query classification result

        Returns:
            Dictionary of retrieval parameters
        """
        query_type = QueryType(classification["query_type"])
        domain = LegalDomain(classification["domain"])
        is_urgent = classification["is_urgent"]
        complexity = classification["complexity"]

        params = {
            "use_hyde": True,
            "use_decomposition": True,
            "k": 5,
            "use_reranking": False,
        }

        if query_type == QueryType.DEFINITION:
            params["k"] = 3
            params["use_decomposition"] = False
            params["use_hyde"] = True

        elif query_type == QueryType.PROCEDURAL:
            params["k"] = 7
            params["use_decomposition"] = True
            params["use_hyde"] = True
            params["use_reranking"] = True

        elif query_type == QueryType.CONDITIONAL:
            params["k"] = 5
            params["use_hyde"] = True
            params["use_decomposition"] = True

        elif query_type == QueryType.TEMPORAL:
            params["k"] = 4
            params["use_decomposition"] = False
            params["use_hyde"] = True

        elif query_type == QueryType.COMPARATIVE:
            params["k"] = 10
            params["use_decomposition"] = True
            params["use_reranking"] = True

        elif query_type == QueryType.CITATION_LOOKUP:
            params["k"] = 3
            params["use_decomposition"] = False
            params["use_hyde"] = False

        if is_urgent:
            params["k"] = max(3, params["k"] - 2)

        if complexity == "complex":
            params["k"] = min(10, params["k"] + 5)
            params["use_decomposition"] = True

        return params

    def route(self, query: str) -> Dict:
        """
        Route a query and get appropriate retrieval parameters.

        Args:
            query: User query

        Returns:
            Dictionary with routing decision and parameters
        """
        classification = self.classifier.classify(query)
        retrieval_params = self.get_retrieval_params(classification)

        return {
            "classification": classification,
            "retrieval_params": retrieval_params,
            "recommended_strategy": self.retrieval_strategies.get(
                QueryType(classification["query_type"]),
                "standard"
            ),
        }


class PromptSelector:
    """Selects appropriate prompts based on query type."""

    def __init__(self):
        """Initialize the prompt selector."""
        self.prompt_templates = {}

    def get_prompt_name(self, query_type: QueryType) -> str:
        """
        Get the appropriate prompt name for a query type.

        Args:
            query_type: Type of query

        Returns:
            Prompt template name
        """
        prompt_mapping = {
            QueryType.DEFINITION: "definition_prompt",
            QueryType.PROCEDURAL: "procedural_prompt",
            QueryType.CONDITIONAL: "conditional_prompt",
            QueryType.RIGHTS_OBLIGATIONS: "rights_prompt",
            QueryType.TEMPORAL: "temporal_prompt",
            QueryType.COMPARATIVE: "comparative_prompt",
            QueryType.HISTORICAL: "historical_prompt",
            QueryType.CITATION_LOOKUP: "citation_prompt",
            QueryType.GENERAL: "general_prompt",
        }
        return prompt_mapping.get(query_type, "general_prompt")


def suggest_follow_up_queries(query_type: QueryType, original_query: str) -> List[str]:
    """
    Suggest follow-up queries based on the original query type.

    Args:
        query_type: Type of the original query
        original_query: Original user query

    Returns:
        List of suggested follow-up queries
    """
    # Pre-process the query to remove common prefixes
    query_clean = original_query.lower().replace("qu'est-ce que ", "")
    
    suggestions = {
        QueryType.DEFINITION: [
            f"Quelles sont les exceptions à {query_clean}?",
            f"Comment appliquer {query_clean} en pratique?",
        ],
        QueryType.PROCEDURAL: [
            "Quels sont les délais à respecter?",
            "Quelles sont les pièces nécessaires?",
            "Qui dois-je contacter?",
        ],
        QueryType.CONDITIONAL: [
            "Y a-t-il des exceptions à ces conditions?",
            "Que se passe-t-il si ces conditions ne sont pas remplies?",
        ],
        QueryType.RIGHTS_OBLIGATIONS: [
            "Quelles sont les sanctions en cas de non-respect?",
            "Comment faire valoir ces droits?",
        ],
        QueryType.TEMPORAL: [
            "Y a-t-il des délais de prescription?",
            "Quand cette règle entre-t-elle en vigueur?",
        ],
        QueryType.COMPARATIVE: [
            "Quelle option est la plus avantageuse?",
            "Quels sont les risques de chaque option?",
        ],
        QueryType.HISTORICAL: [
            "Pourquoi cette loi a-t-elle été modifiée?",
            "Quelle était la situation avant cette réforme?",
        ],
        QueryType.CITATION_LOOKUP: [
            "Où trouver le texte complet de cette loi?",
            "Y a-t-il des décrets d'application?",
        ],
        QueryType.GENERAL: [
            "Pouvez-vous donner plus de détails?",
            "Quelles sont les implications pratiques?",
        ],
    }

    return suggestions.get(query_type, suggestions[QueryType.GENERAL])


if __name__ == "__main__":
    classifier = QueryClassifier()
    router = QueryRouter()

    test_queries = [
        "Qu'est-ce que la période d'essai en droit du travail?",
        "Comment rompre un contrat de travail à durée déterminée?",
        "Quelles sont les conditions pour demander un congé parental?",
        "Quel est le délai de prescription pour un accident du travail?",
        "Quelle est la différence entre un CDI et un CDD?",
        "Article L1234-1 du Code du travail",
        "Comment rédiger une rupture conventionnelle?",
    ]

    print("Query Classification Results:")
    print("=" * 70)

    for query in test_queries:
        result = router.route(query)
        print(f"\nQuery: {query}")
        print(f"  Type: {result['classification']['query_type']} "
              f"(conf: {result['classification']['query_type_confidence']:.2f})")
        print(f"  Domain: {result['classification']['domain']} "
              f"(conf: {result['classification']['domain_confidence']:.2f})")
        print(f"  Complexity: {result['classification']['complexity']}")
        print(f"  Retrieval k: {result['retrieval_params']['k']}")
        print(f"  Use HyDE: {result['retrieval_params']['use_hyde']}")
        print(f"  Use Decomposition: {result['retrieval_params']['use_decomposition']}")
