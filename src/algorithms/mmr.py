from typing import Any, Dict, List, Tuple

from .base import BaseDiversifier
from .uvctr import UVCTRRanker


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _to_token_set(tokens) -> frozenset:
    """Normalise any token representation to a frozenset of ints."""
    if tokens is None:
        return frozenset()
    if isinstance(tokens, (list, tuple)):
        if not tokens:
            return frozenset()
        # Might be list-of-ints or list-of-strings
        result = []
        for t in tokens:
            if isinstance(t, str):
                result.extend(int(x) for x in t.split(",") if x.strip())
            else:
                result.append(int(t))
        return frozenset(result)
    if isinstance(tokens, str):
        return frozenset(int(x) for x in tokens.split(",") if x.strip())
    return frozenset()


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


# ---------------------------------------------------------------------------
# MMR ranker
# ---------------------------------------------------------------------------

# product_text field index for each metadata type
_FIELD_IDX = {
    "name": 0,
    "cate": 1,
    "brand": 2,
    "size": 3,
    "attr": 4,
    "color": 5,
}


class MMRRanker(BaseDiversifier):
    """
    Maximal Marginal Relevance (MMR) diversification.

    At each greedy step, selects the unranked product that maximises:
        lambda_ * rel(p) - (1 - lambda_) * max_{s in selected} sim(p, s)

    Parameters
    ----------
    relevance : str
        'rel'   – use raw relevance score from the platform model.
        'uvctr' – use the UVCTR combined score (relevance + popularity).
    feature : str
        Metadata field to use for the pairwise similarity term.
        One of 'name', 'cate', 'brand', 'size', 'attr', 'color'.
    lambda_ : float
        Trade-off weight: 1.0 = pure relevance, 0.0 = pure novelty.
        The original MMR paper uses 0.5.
    """

    def __init__(
        self,
        relevance: str = "rel",
        feature: str = "name",
        lambda_: float = 0.5,
    ):
        assert relevance in ("rel", "uvctr"), "relevance must be 'rel' or 'uvctr'"
        assert feature in _FIELD_IDX, f"feature must be one of {list(_FIELD_IDX)}"
        self.relevance = relevance
        self.feature = feature
        self.lambda_ = lambda_
        self._uvctr = UVCTRRanker()

    # ------------------------------------------------------------------
    def _rel_scores(
        self, initial_ranking: List[Tuple[Any, float]], context: Dict
    ) -> Dict[Any, float]:
        if self.relevance == "uvctr":
            return {
                pid: self._uvctr.score(pid, rel, context)
                for pid, rel in initial_ranking
            }
        return {pid: rel for pid, rel in initial_ranking}

    def _feature_tokens(self, pid, context: Dict) -> frozenset:
        text = context["product_text"].get(pid)
        if text is None:
            return frozenset()
        idx = _FIELD_IDX[self.feature]
        if idx >= len(text):
            return frozenset()
        return _to_token_set(text[idx])

    # ------------------------------------------------------------------
    def rerank(self, query, initial_ranking, context):
        rel_scores = self._rel_scores(initial_ranking, context)
        candidates = [pid for pid, _ in initial_ranking]

        # Pre-compute feature token sets
        token_sets = {pid: self._feature_tokens(pid, context) for pid in candidates}

        selected: List[Any] = []
        selected_tokens: List[frozenset] = []
        remaining = list(candidates)

        while remaining:
            best_pid, best_score = None, float("-inf")
            for pid in remaining:
                rel = rel_scores.get(pid, 0.0)
                if selected_tokens:
                    max_sim = max(jaccard(token_sets[pid], st) for st in selected_tokens)
                else:
                    max_sim = 0.0
                score = self.lambda_ * rel - (1 - self.lambda_) * max_sim
                if score > best_score:
                    best_score = score
                    best_pid = pid

            selected.append(best_pid)
            selected_tokens.append(token_sets[best_pid])
            remaining.remove(best_pid)

        return selected

    def __repr__(self) -> str:
        return f"MMR(rel={self.relevance}, feat={self.feature}, λ={self.lambda_})"
