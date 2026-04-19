from .base import BaseDiversifier


class RELRanker(BaseDiversifier):
    """
    REL baseline: rank purely by the platform relevance score.
    No diversification is applied.
    """

    def rerank(self, query, initial_ranking, context):
        # initial_ranking is already sorted descending by relevance score
        return [pid for pid, _ in initial_ranking]
