"""
JDivPS data loader.

Handles the five compressed-pickle files and the two CSV annotation files
that make up the JDivPS dataset.  All text is stored as anonymised integer
token IDs (the platform's private tokeniser).

Data files expected in `data_dir`
----------------------------------
dict_product_text_release.pkl.gz
    {product_id: [name_toks, cate_toks, brand_toks,
                  size_toks, attr_toks, color_toks]}

product_uvctr_dict_release.pkl.gz
    {product_id: [uv, pv, ctr]}   (uv/pv in 0-10 000, ctr in 0-1)

query_suggestions_release.pkl.gz
    {query: [suggestion, ...]}   (up to 10 per query)

query_product_features_release.pkl.gz
    {(query, product_id): [rel, tfidf_name, tfidf_cate, tfidf_brand,
                            bm25_name, bm25_cate, bm25_brand, uv, pv, ctr]}

query_intent_label_ts.csv   (tab-separated, test set, human-annotated)
query_intent_label_tr.csv   (tab-separated, training set, model-annotated)
    Columns: query \\t intent \\t product_id \\t label
    Queries and intents are stored as comma-separated integer token IDs.
    Only rows with label=1 (positive) are loaded.
"""

import gzip
import os
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _load_pkl_gz(path: str):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _norm_query(q) -> tuple:
    """Normalise a query representation to a tuple of ints."""
    if isinstance(q, tuple):
        return q
    if isinstance(q, list):
        return tuple(q)
    if isinstance(q, str):
        return tuple(int(x) for x in q.split(",") if x.strip())
    # single int token
    return (int(q),)


def _norm_tokens(tokens) -> List[int]:
    """Normalise any token field to a flat list of ints."""
    if tokens is None:
        return []
    if isinstance(tokens, (list, tuple)):
        result = []
        for t in tokens:
            if isinstance(t, str):
                result.extend(int(x) for x in t.split(",") if x.strip())
            elif t is not None:
                result.append(int(t))
        return result
    if isinstance(tokens, str):
        return [int(x) for x in tokens.split(",") if x.strip()]
    return []


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class JDivPSLoader:
    """
    Lazy-loading wrapper for the JDivPS dataset files.

    After construction call `get_test_data()` (or `get_train_data()`) to
    obtain a list of per-query dicts ready for the evaluator.
    """

    _FILES = {
        "product_text":     "dict_product_text_release.pkl.gz",
        "product_uvctr":    "product_uvctr_dict_release.pkl.gz",
        "suggestions":      "query_suggestions_release.pkl.gz",
        "features":         "query_product_features_release.pkl.gz",
        "labels_test":      "query_intent_label_ts.csv",
        "labels_train":     "query_intent_label_tr.csv",
    }

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._cache: Dict[str, Any] = {}

    def _path(self, key: str) -> str:
        return os.path.join(self.data_dir, self._FILES[key])

    def _require(self, key: str):
        """Load and cache a dataset file."""
        if key not in self._cache:
            path = self._path(key)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"JDivPS file not found: {path}\n"
                    f"Download it from https://github.com/DengZhirui/JDivPS "
                    f"and place it in: {self.data_dir}"
                )
            print(f"Loading {self._FILES[key]} …", flush=True)
            if key.startswith("labels"):
                self._cache[key] = pd.read_csv(path, sep="\t", dtype=str)
            else:
                self._cache[key] = _load_pkl_gz(path)
        return self._cache[key]

    # ------------------------------------------------------------------
    # Public accessors (raw data)
    # ------------------------------------------------------------------

    @property
    def product_text(self) -> Dict:
        return self._require("product_text")

    @property
    def product_uvctr(self) -> Dict:
        return self._require("product_uvctr")

    @property
    def raw_suggestions(self) -> Dict:
        return self._require("suggestions")

    @property
    def raw_features(self) -> Dict:
        return self._require("features")

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_feature_index(self) -> Dict[tuple, Dict]:
        """
        Group query_product_features by query.
        Returns: {query_tuple: {product_id: feature_list}}
        """
        if "_feature_index" not in self._cache:
            index: Dict[tuple, Dict] = defaultdict(dict)
            for (q, pid), feats in self.raw_features.items():
                index[_norm_query(q)][pid] = feats
            self._cache["_feature_index"] = dict(index)
        return self._cache["_feature_index"]

    def _build_suggestion_index(self) -> Dict[tuple, List]:
        """Returns: {query_tuple: [suggestion_tuple, ...]}"""
        if "_sugg_index" not in self._cache:
            index: Dict[tuple, List] = {}
            for q, suggs in self.raw_suggestions.items():
                nq = _norm_query(q)
                index[nq] = [_norm_query(s) for s in suggs]
            self._cache["_sugg_index"] = index
        return self._cache["_sugg_index"]

    def _load_intent_labels(self, split: str) -> Dict[tuple, Dict[tuple, set]]:
        """
        Load intent annotation CSV.

        Returns:
            {query_tuple: {intent_tuple: set(product_ids)}}
        """
        key = f"labels_{split}"
        df = self._require(key)

        # Column names may vary; normalise
        col_map = {}
        for col in df.columns:
            lc = col.strip().lower()
            if "query" in lc and "suggest" not in lc:
                col_map["query"] = col
            elif "intent" in lc:
                col_map["intent"] = col
            elif "product" in lc or lc == "doc":
                col_map["product_id"] = col
            elif "label" in lc or "relation" in lc:
                col_map["label"] = col

        missing = [k for k in ("query", "intent", "product_id", "label") if k not in col_map]
        if missing:
            raise ValueError(
                f"Could not find columns {missing} in {self._FILES[key]}.\n"
                f"Actual columns: {list(df.columns)}"
            )

        # Keep only positive labels
        pos = df[df[col_map["label"]].str.strip() == "1"]

        result: Dict[tuple, Dict[tuple, set]] = defaultdict(lambda: defaultdict(set))
        for _, row in pos.iterrows():
            q = _norm_query(row[col_map["query"]])
            intent = _norm_query(row[col_map["intent"]])
            pid = row[col_map["product_id"]].strip()
            # Try to match product_id type to what's used in features dict
            result[q][intent].add(pid)

        return {q: dict(imap) for q, imap in result.items()}

    # ------------------------------------------------------------------
    # Build per-query data dicts for the evaluator
    # ------------------------------------------------------------------

    def _build_query_data(self, split: str) -> List[Dict]:
        """
        Build a list of per-query dicts:
        {
            'query':              tuple[int, ...],
            'initial_ranking':    [(pid, rel_score), ...],  # 200 items, desc rel
            'intent_product_map': {intent_tuple: set(pids)},
            'context': {
                'product_text':  {pid: [name_toks, ...]},
                'product_uvctr': {pid: [uv, pv, ctr]},
                'features':      {pid: feature_list},
                'suggestions':   [suggestion_tuple, ...],
            }
        }
        """
        intent_labels  = self._load_intent_labels(split)
        feature_index  = self._build_feature_index()
        sugg_index     = self._build_suggestion_index()
        prod_text      = self.product_text
        prod_uvctr     = self.product_uvctr

        query_data = []
        skipped = 0

        for query, intent_map in intent_labels.items():
            pid_features = feature_index.get(query)
            if not pid_features:
                skipped += 1
                continue

            # Build initial ranking sorted by relevance score (index 0)
            # Store (pid, rel_score) — just the scalar, not the full feature list
            initial_ranking = sorted(
                [(pid, feats[0]) for pid, feats in pid_features.items()],
                key=lambda kv: kv[1],
                reverse=True,
            )

            # Resolve product_id types: the intent CSV stores pids as strings,
            # but the features dict may use strings or ints.  Try to reconcile.
            # (If no type conversion needed, the set intersection will still work.)
            ipm = {intent: set(pids) for intent, pids in intent_map.items()}

            context = {
                "product_text":  {pid: prod_text.get(pid, []) for pid, _ in initial_ranking},
                "product_uvctr": {pid: prod_uvctr.get(pid, [0, 0, 0]) for pid, _ in initial_ranking},
                "features":      {pid: feats for pid, feats in pid_features.items()},
                "suggestions":   sugg_index.get(query, []),
            }

            query_data.append({
                "query":              query,
                "initial_ranking":    initial_ranking,
                "intent_product_map": ipm,
                "context":            context,
            })

        if skipped:
            print(
                f"  [warn] {skipped} queries from {split} labels had no matching "
                f"features entry and were skipped."
            )
        print(f"  Loaded {len(query_data)} {split} queries.")
        return query_data

    def get_test_data(self) -> List[Dict]:
        """Return per-query evaluation dicts for the human-annotated test set."""
        return self._build_query_data("test")

    def get_train_data(self) -> List[Dict]:
        """Return per-query evaluation dicts for the (model-annotated) training set."""
        return self._build_query_data("train")
