"""
Diversity evaluation metrics used in JDivPS / TREC Web Track.

All functions take:
    ranked_products     : list of product_ids in ranking order (best first)
    intent_product_map  : dict  { intent_key → set(product_ids) }
                          intent_key can be any hashable (e.g. tuple of ints)

References
----------
Clarke et al. 2008 – α-nDCG
Chapelle et al. 2009 – ERR-IA
Clarke et al. 2009 – NRBP
Zhai et al. 2015 – S-rec (Subtopic Recall)
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_pid_intents(intent_product_map: Dict) -> Dict:
    """Reverse index: product_id → set of intents it covers."""
    pid_intents: Dict[Any, set] = defaultdict(set)
    for intent, prod_set in intent_product_map.items():
        for pid in prod_set:
            pid_intents[pid].add(intent)
    return pid_intents


def _compute_idcg(
    pid_intents: Dict,
    candidates: List,
    alpha: float,
    k: int,
) -> float:
    """
    Greedy ideal α-DCG over the candidate pool.

    At each position we greedily pick the product that yields the
    highest marginal gain given what has already been selected.
    Only considers products that cover ≥1 intent (others contribute 0).
    """
    # Filter to products with any intent coverage
    relevant = [pid for pid in candidates if pid_intents.get(pid)]
    remaining = set(relevant)
    intent_counts: Dict[Any, int] = defaultdict(int)
    idcg = 0.0

    for i in range(min(k, len(remaining))):
        best_gain = -1.0
        best_pid = None
        for pid in remaining:
            gain = sum(
                (1 - alpha) ** intent_counts[intent]
                for intent in pid_intents[pid]
            )
            if gain > best_gain:
                best_gain = gain
                best_pid = pid

        if best_pid is None or best_gain <= 0:
            break

        idcg += best_gain / math.log2(i + 2)   # rank i+1, discount = 1/log2(rank+1)
        for intent in pid_intents[best_pid]:
            intent_counts[intent] += 1
        remaining.discard(best_pid)

    return idcg


# ---------------------------------------------------------------------------
# α-nDCG  (Clarke et al. 2008)
# ---------------------------------------------------------------------------

def alpha_ndcg(
    ranked_products: List,
    intent_product_map: Dict,
    alpha: float = 0.5,
    k: int = 10,
) -> float:
    """
    α-nDCG@K.

    Penalises repeated coverage of the same intent by (1-α) per repeat.
    α=0 → no penalty (= standard nDCG over intents),
    α=1 → hard novelty (only first hit counts).
    """
    if not intent_product_map:
        return 0.0

    pid_intents = _build_pid_intents(intent_product_map)

    # --- DCG ---
    intent_counts: Dict[Any, int] = defaultdict(int)
    dcg = 0.0
    for i, pid in enumerate(ranked_products[:k]):
        gain = sum(
            (1 - alpha) ** intent_counts[intent]
            for intent in pid_intents.get(pid, set())
        )
        dcg += gain / math.log2(i + 2)
        for intent in pid_intents.get(pid, set()):
            intent_counts[intent] += 1

    # --- IDCG (greedy over the candidate pool = the full ranked list) ---
    idcg = _compute_idcg(pid_intents, list(ranked_products), alpha, k)

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# ERR-IA  (Chapelle et al. 2009, intent-aware variant)
# ---------------------------------------------------------------------------

def err_ia(
    ranked_products: List,
    intent_product_map: Dict,
    k: int = 10,
) -> float:
    """
    ERR-IA@K  (uniform intent prior = 1/|I|).

    ERR-IA = (1/|I|) * Σ_j  Σ_{i=1}^{K}  R_{ij} · Π_{l<i}(1-R_{lj}) · (1/i)
    """
    if not intent_product_map:
        return 0.0

    n_intents = len(intent_product_map)
    total = 0.0

    for relevant in intent_product_map.values():
        p_not_found = 1.0
        for i, pid in enumerate(ranked_products[:k]):
            r = 1.0 if pid in relevant else 0.0
            total += p_not_found * r / (i + 1)
            p_not_found *= 1.0 - r
            if p_not_found < 1e-12:
                break

    return total / n_intents


# ---------------------------------------------------------------------------
# P-IA  (Intent-Aware Precision)
# ---------------------------------------------------------------------------

def p_ia(
    ranked_products: List,
    intent_product_map: Dict,
    k: int = 10,
) -> float:
    """
    P-IA@K = (1/|I|) · Σ_j  (# relevant docs in top-K for intent j) / K
    """
    if not intent_product_map:
        return 0.0

    top_k = ranked_products[:k]
    n_intents = len(intent_product_map)
    total = 0.0

    for relevant in intent_product_map.values():
        hits = sum(1 for pid in top_k if pid in relevant)
        total += hits / k

    return total / n_intents


# ---------------------------------------------------------------------------
# S-rec  (Subtopic Recall / Zhai et al. 2015)
# ---------------------------------------------------------------------------

def s_rec(
    ranked_products: List,
    intent_product_map: Dict,
    k: int = 10,
) -> float:
    """
    S-rec@K = (1/|I|) · |{ j : ∃ d ∈ top-K that is relevant to j }|

    The fraction of intents covered at least once in the top-K results.
    """
    if not intent_product_map:
        return 0.0

    top_k_set = set(ranked_products[:k])
    covered = sum(
        1 for relevant in intent_product_map.values()
        if top_k_set & relevant
    )
    return covered / len(intent_product_map)


# ---------------------------------------------------------------------------
# NRBP  (Clarke et al. 2009)
# ---------------------------------------------------------------------------

def nrbp(
    ranked_products: List,
    intent_product_map: Dict,
    beta: float = 0.95,
    gamma: float = 0.5,
) -> float:
    """
    NRBP (Novelty and Relevance-aware Browsing Probability).

    NRBP = (1/|I|) · Σ_j  Σ_i  p(user at rank i for intent j) · R_{ij}

    where p(at i, j) = β^{i-1} · Π_{l<i}(1 - γ · R_{lj})

    β  controls patience (how likely the user continues browsing).
    γ  controls how much a relevant document reduces further browsing.
    """
    if not intent_product_map:
        return 0.0

    n_intents = len(intent_product_map)
    total = 0.0

    for relevant in intent_product_map.values():
        p_at_i = 1.0
        for pid in ranked_products:
            r = 1.0 if pid in relevant else 0.0
            total += p_at_i * r
            p_at_i *= beta * (1.0 - gamma * r)
            if p_at_i < 1e-12:
                break

    return total / n_intents


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_all_metrics(
    ranked_products: List,
    intent_product_map: Dict,
    ks: Tuple[int, ...] = (10, 20),
    alpha: float = 0.5,
    beta: float = 0.95,
    gamma: float = 0.5,
) -> Dict[str, float]:
    """
    Compute every metric for one query and return a flat dict.

    Keys follow the paper's notation, e.g. 'alpha_ndcg@10', 'err_ia@20',
    'nrbp', 'p_ia@10', 's_rec@20'.
    """
    # Ensure intent sets are proper Python sets for O(1) lookup
    ipm = {k: set(v) for k, v in intent_product_map.items()}

    results: Dict[str, float] = {}
    for k in ks:
        results[f"alpha_ndcg@{k}"] = alpha_ndcg(ranked_products, ipm, alpha=alpha, k=k)
        results[f"err_ia@{k}"]     = err_ia(ranked_products, ipm, k=k)
        results[f"p_ia@{k}"]       = p_ia(ranked_products, ipm, k=k)
        results[f"s_rec@{k}"]      = s_rec(ranked_products, ipm, k=k)

    results["nrbp"] = nrbp(ranked_products, ipm, beta=beta, gamma=gamma)
    return results
