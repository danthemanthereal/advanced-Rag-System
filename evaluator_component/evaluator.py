import math

class Evaluator:

    def __init__(self,):
        pass

    def ndcg_at_k(self, predicted_ids: list[str], relevant: dict[str, int], k: int = 10):
        dcg = sum(
            relevant.get(doc_id, 0) / math.log2(rank + 2)
            for rank, doc_id in enumerate(predicted_ids[:k])
        )
        print("ids ", predicted_ids)
        print("rel ", relevant)
        ideal_rels = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_rels))
        return dcg / idcg if idcg > 0 else 0.0