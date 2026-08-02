from collections import defaultdict


class ReciprocalRankFusion:

    def __init__(self,k_rrf):
        self.k_rrf= k_rrf

    def merge_with_rrf(self, rankings: list[list[str]])-> list[tuple[str, float]]:
        scores: dict[str, float] = defaultdict(float)
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                scores[doc_id] += 1.0 / (self.k_rrf + rank)
        return sorted(scores.items(), key=lambda x: -x[1])

