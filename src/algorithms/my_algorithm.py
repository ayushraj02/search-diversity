"""
============================================================
  YOUR DIVERSIFICATION ALGORITHM
============================================================

Edit the `rerank` method of MyDiversificationAlgorithm below.
The scaffold already wires it into the evaluation pipeline —
running `run_eval.py` will benchmark it against all baselines.

What you receive
----------------
query : tuple[int, ...]
    Anonymised token IDs of the query (e.g. (1492, 307, 88)).

initial_ranking : list[tuple[product_id, float]]
    200 (product_id, relevance_score) pairs, sorted desc by the
    JD.com platform relevance model.  relevance_score ∈ [0, 1].

context : dict
    ├─ 'product_text'  → dict[pid → list]
    │      [name_toks, cate_toks, brand_toks,
    │       size_toks,  attr_toks,  color_toks]
    │      Each element is a list[int] of anonymised token IDs.
    │
    ├─ 'product_uvctr' → dict[pid → list[float]]
    │      [uv, pv, ctr]
    │      uv and pv are in 0-10 000 range; ctr is 0-1.
    │
    ├─ 'features'      → dict[pid → list[float]]
    │      [relevance, tfidf_name, tfidf_cate, tfidf_brand,
    │       bm25_name,  bm25_cate,  bm25_brand, uv, pv, ctr]
    │
    └─ 'suggestions'   → list[tuple[int, ...]]
           Up to 10 query suggestions from JD.com used as intent
           proxies at inference time (not ground-truth intents).

What you return
---------------
A list of ALL product_ids from initial_ranking in your preferred
order (best → worst).  The evaluator will truncate at K=10 / K=20.
============================================================
"""

from typing import Any, Dict, List, Tuple

from .base import BaseDiversifier


class MyDiversificationAlgorithm(BaseDiversifier):
    """Plug your diversification technique in here."""

    def __init__(self):
        # Initialise any hyperparameters or models you need.
        pass

    def rerank(
        self,
        query: tuple,
        initial_ranking: List[Tuple[Any, float]],
        context: Dict[str, Any],
    ) -> List[Any]:
        """
        Replace the body of this method with your algorithm.

        The default below is a simple relevance passthrough so the
        code runs out of the box before you implement anything.
        """

        # ==============================================================
        # IMPLEMENT YOUR ALGORITHM BELOW
        # ==============================================================

        selected = []
        remaining = list(initial_ranking)   # list of (pid, rel_score)

        while remaining:
            # ---------------------------------------------------------
            # Example: pick the next product greedily.
            # Replace this block with your selection logic.
            # ---------------------------------------------------------
            best_pid = remaining[0][0]      # <-- your criterion here
            selected.append(best_pid)
            remaining = [(p, s) for p, s in remaining if p != best_pid]

        return selected

        # ==============================================================
        # QUICK-START SNIPPETS (delete once you have your own logic)
        # ==============================================================
        #
        # -- Access relevance scores:
        #   rel_score = dict(initial_ranking)[pid]   # or context['features'][pid][0]
        #
        # -- Access product tokens (for overlap / similarity):
        #   name_toks  = context['product_text'][pid][0]
        #   brand_toks = context['product_text'][pid][2]
        #   cate_toks  = context['product_text'][pid][1]
        #
        # -- Access popularity:
        #   uv, pv, ctr = context['product_uvctr'][pid]
        #
        # -- Access query suggestions (intent proxies):
        #   for sugg_toks in context['suggestions']:
        #       overlap = set(sugg_toks) & set(name_toks)
        #
        # ==============================================================
