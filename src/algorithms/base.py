from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class BaseDiversifier(ABC):
    """Abstract base class for all diversification algorithms."""

    @abstractmethod
    def rerank(
        self,
        query: tuple,
        initial_ranking: List[Tuple[Any, float]],
        context: Dict[str, Any],
    ) -> List[Any]:
        """
        Rerank products for diversity.

        Args:
            query:
                Query as a tuple of integer token IDs, e.g. (123, 456, 789).
                All text in JDivPS is anonymized to integer IDs.

            initial_ranking:
                List of (product_id, relevance_score) pairs, sorted descending
                by the platform relevance model score.  200 products per query.

            context:
                Dict with the following keys:

                'product_text'  → dict[pid, list]
                    [name_tokens, category_tokens, brand_tokens,
                     size_tokens, attribute_tokens, color_tokens]
                    Each element is a list of int token IDs (may be empty []).

                'product_uvctr' → dict[pid, list]
                    [uv, pv, ctr]  (uv/pv: 0-10 000 float, ctr: 0-1 float)

                'features'      → dict[pid, list]
                    [relevance_score, tfidf_name, tfidf_category, tfidf_brand,
                     bm25_name, bm25_category, bm25_brand, uv, pv, ctr]

                'suggestions'   → list[tuple[int, ...]]
                    Query suggestions (platform subtopics) used as intent
                    proxies at inference time; one list per query, 10 entries.

        Returns:
            List of product_ids in your desired ranking order (best first).
            Must contain all product_ids from initial_ranking.
        """
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__
