from .base import BaseDiversifier


class UVCTRRanker(BaseDiversifier):
    """
    UVCTR baseline: combines relevance with product popularity.

    Score formula (from JDivPS paper, §5.1.1):
        S_{q,p} = (Rel(s_{q,p}) * 10_000 + UV_p * CTR_p) / 50_000

    where Rel(·) maps the continuous relevance score into a 5-level
    integer label 0–4, and UV_p is already on a 0–10 000 scale.
    The 10 000 multiplier ensures any relevant product always outscores
    any irrelevant one regardless of popularity.

    This is a re-ranking baseline (not an initial retrieval baseline).
    """

    N_LEVELS = 5  # relevance levels 0-4

    def rel_to_level(self, rel_score: float) -> int:
        """Map continuous [0, 1] relevance score to integer level 0–4."""
        return min(int(rel_score * self.N_LEVELS), self.N_LEVELS - 1)

    def score(self, pid, rel_score: float, context) -> float:
        feat = context["features"].get(pid)
        if feat is not None:
            uv, ctr = feat[7], feat[9]
        else:
            uv, ctr = 0.0, 0.0
        rel_level = self.rel_to_level(rel_score)
        return (rel_level * 10_000 + uv * ctr) / 50_000.0

    def rerank(self, query, initial_ranking, context):
        scored = [
            (pid, self.score(pid, rel, context))
            for pid, rel in initial_ranking
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in scored]
