"""
Evaluator: run a set of diversification algorithms over JDivPS test queries
and print a metrics comparison table identical to Table 5 in the paper.
"""

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .metrics import compute_all_metrics


# Metric display order (matches Table 5 in the JDivPS paper)
_METRIC_COLS = [
    "alpha_ndcg@10",
    "alpha_ndcg@20",
    "err_ia@10",
    "err_ia@20",
    "nrbp",
    "p_ia@10",
    "p_ia@20",
    "s_rec@10",
    "s_rec@20",
]

_DISPLAY_NAMES = {
    "alpha_ndcg@10": "α-nDCG@10",
    "alpha_ndcg@20": "α-nDCG@20",
    "err_ia@10":     "ERR-IA@10",
    "err_ia@20":     "ERR-IA@20",
    "nrbp":          "NRBP",
    "p_ia@10":       "P-IA@10",
    "p_ia@20":       "P-IA@20",
    "s_rec@10":      "S-rec@10",
    "s_rec@20":      "S-rec@20",
}


class Evaluator:
    """
    Run diversification algorithms on JDivPS and report aggregated metrics.

    Parameters
    ----------
    ks : tuple[int, ...]
        Cut-off depths for metrics.  Default (10, 20) matches the paper.
    alpha : float
        α penalty in α-nDCG.
    beta, gamma : float
        Patience and novelty parameters for NRBP.
    """

    def __init__(
        self,
        ks: tuple = (10, 20),
        alpha: float = 0.5,
        beta: float = 0.95,
        gamma: float = 0.5,
    ):
        self.ks = ks
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    # ------------------------------------------------------------------
    def evaluate_one(
        self,
        algorithm,
        query_data: Dict,
    ) -> Dict[str, float]:
        """
        Run `algorithm` on a single query dict and return metric scores.

        query_data keys: 'query', 'initial_ranking', 'intent_product_map', 'context'
        """
        ranked = algorithm.rerank(
            query_data["query"],
            query_data["initial_ranking"],
            query_data["context"],
        )
        return compute_all_metrics(
            ranked,
            query_data["intent_product_map"],
            ks=self.ks,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )

    # ------------------------------------------------------------------
    def evaluate_algorithm(
        self,
        name: str,
        algorithm,
        test_data: List[Dict],
        show_progress: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate one algorithm over all test queries.

        Returns macro-averaged metric scores.
        """
        accum: Dict[str, float] = defaultdict(float)
        iterator = tqdm(test_data, desc=f"  {name}", leave=False) if show_progress else test_data

        for qd in iterator:
            scores = self.evaluate_one(algorithm, qd)
            for metric, val in scores.items():
                accum[metric] += val

        n = len(test_data)
        return {m: v / n for m, v in accum.items()}

    # ------------------------------------------------------------------
    def evaluate_all(
        self,
        test_data: List[Dict],
        algorithms: Dict[str, Any],
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate every algorithm in `algorithms`.

        Parameters
        ----------
        algorithms : dict  { display_name → BaseDiversifier instance }

        Returns
        -------
        results : dict  { display_name → { metric → score } }
        """
        results: Dict[str, Dict[str, float]] = {}
        for name, algo in algorithms.items():
            print(f"Evaluating: {name}")
            t0 = time.time()
            results[name] = self.evaluate_algorithm(name, algo, test_data, show_progress)
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")
        return results

    # ------------------------------------------------------------------
    def print_results(
        self,
        results: Dict[str, Dict[str, float]],
        highlight_best: bool = True,
    ) -> None:
        """
        Print a formatted metrics table matching Table 5 of the JDivPS paper.

        Best values in each column are highlighted with *.
        """
        try:
            from tabulate import tabulate
            _have_tabulate = True
        except ImportError:
            _have_tabulate = False

        metrics = [m for m in _METRIC_COLS if m in next(iter(results.values()))]
        headers = ["Model"] + [_DISPLAY_NAMES[m] for m in metrics]

        # Find best per column
        best: Dict[str, float] = {}
        if highlight_best:
            for m in metrics:
                best[m] = max(scores[m] for scores in results.values())

        rows = []
        for name, scores in results.items():
            row = [name]
            for m in metrics:
                val = scores.get(m, 0.0)
                cell = f"{val:.4f}"
                if highlight_best and abs(val - best.get(m, -1)) < 1e-9:
                    cell = f"*{cell}"
                row.append(cell)
            rows.append(row)

        print()
        print("=" * 10, "Results", "=" * 10)
        if _have_tabulate:
            print(tabulate(rows, headers=headers, tablefmt="github"))
        else:
            # Fallback plain text table
            col_w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
            sep = "  ".join("-" * w for w in col_w)
            fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
            print(fmt.format(*headers))
            print(sep)
            for row in rows:
                print(fmt.format(*row))
        print()

    def save_results(
        self,
        results: Dict[str, Dict[str, float]],
        path: str,
    ) -> None:
        """Save results to a CSV file."""
        import csv
        metrics = [m for m in _METRIC_COLS if m in next(iter(results.values()))]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model"] + metrics)
            for name, scores in results.items():
                writer.writerow([name] + [f"{scores.get(m, 0.0):.6f}" for m in metrics])
        print(f"Results saved to {path}")
