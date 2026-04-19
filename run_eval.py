#!/usr/bin/env python3
"""
JDivPS diversity evaluation framework
======================================

Quick start
-----------
1. Place the JDivPS data files in `data/` (see data/.gitkeep for the list).
2. Implement your algorithm in src/algorithms/my_algorithm.py.
3. Run:
       python run_eval.py

Options
-------
  --data-dir DIR        Directory containing JDivPS files. Default: data/
  --output CSV          Save results table to a CSV file.
  --no-my-algo          Skip MyDiversificationAlgorithm (useful before you
                        implement it, so it doesn't pollute the table).
  --alpha FLOAT         α parameter for α-nDCG (default 0.5).
  --lambda FLOAT        λ parameter for MMR (default 0.5).
  --split {test,train}  Which annotation split to evaluate on (default test).
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import JDivPSLoader
from src.evaluator import Evaluator
from src.algorithms import (
    RELRanker,
    UVCTRRanker,
    MMRRanker,
    MyDiversificationAlgorithm,
)


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

def build_algorithms(lambda_: float, include_my_algo: bool) -> dict:
    """
    Return an ordered dict of { display_name → algorithm_instance }.

    Add or remove algorithms here to change what gets benchmarked.
    """
    algos = {
        # ---- Non-diversified baselines ----
        "REL":              RELRanker(),
        "UVCTR":            UVCTRRanker(),

        # ---- MMR with REL relevance ----
        "MMR_REL_name":     MMRRanker(relevance="rel",   feature="name",  lambda_=lambda_),
        "MMR_REL_brand":    MMRRanker(relevance="rel",   feature="brand", lambda_=lambda_),
        "MMR_REL_cate":     MMRRanker(relevance="rel",   feature="cate",  lambda_=lambda_),

        # ---- MMR with UVCTR relevance ----
        "MMR_UVCTR_name":   MMRRanker(relevance="uvctr", feature="name",  lambda_=lambda_),
        "MMR_UVCTR_brand":  MMRRanker(relevance="uvctr", feature="brand", lambda_=lambda_),
        "MMR_UVCTR_cate":   MMRRanker(relevance="uvctr", feature="cate",  lambda_=lambda_),
    }

    if include_my_algo:
        # ----------------------------------------------------------------
        # YOUR ALGORITHM
        # Edit src/algorithms/my_algorithm.py to implement it.
        # ----------------------------------------------------------------
        algos["MyAlgorithm"] = MyDiversificationAlgorithm()

    return algos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate diversification algorithms on JDivPS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir",    default="data",  help="Path to JDivPS data files")
    parser.add_argument("--output",      default=None,    help="Save results to this CSV path")
    parser.add_argument("--no-my-algo",  action="store_true", help="Skip MyDiversificationAlgorithm")
    parser.add_argument("--alpha",       type=float, default=0.5,  help="α for α-nDCG")
    parser.add_argument("--lambda",      type=float, default=0.5,  dest="lambda_", help="λ for MMR")
    parser.add_argument("--beta",        type=float, default=0.95, help="β for NRBP (patience)")
    parser.add_argument("--gamma",       type=float, default=0.5,  help="γ for NRBP (novelty)")
    parser.add_argument("--split",       default="test", choices=["test", "train"],
                        help="Annotation split to evaluate on")
    args = parser.parse_args()

    # Resolve data directory relative to this script's location
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)

    print(f"\nJDivPS Diversity Evaluation")
    print(f"  data dir : {args.data_dir}")
    print(f"  split    : {args.split}")
    print(f"  α (nDCG) : {args.alpha}")
    print(f"  λ (MMR)  : {args.lambda_}")
    print()

    # 1. Load data
    loader = JDivPSLoader(args.data_dir)
    if args.split == "test":
        test_data = loader.get_test_data()
    else:
        test_data = loader.get_train_data()

    if not test_data:
        print("No query data loaded. Check your data directory and file names.")
        sys.exit(1)

    print(f"\nEvaluating {len(test_data)} queries …\n")

    # 2. Build algorithm set
    algorithms = build_algorithms(
        lambda_=args.lambda_,
        include_my_algo=not args.no_my_algo,
    )

    # 3. Run evaluation
    evaluator = Evaluator(
        ks=(10, 20),
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )
    results = evaluator.evaluate_all(test_data, algorithms)

    # 4. Display results
    evaluator.print_results(results)

    # 5. Optionally save
    if args.output:
        evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()
