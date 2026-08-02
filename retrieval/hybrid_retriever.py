from reciprocal_rank_fusion.rrf import ReciprocalRankFusion
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever


class HybridRetriever:

    def __init__(self, bm25_retriever: BM25Retriever,
                 dense_retriever: DenseRetriever,
                 rrf: ReciprocalRankFusion ):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf=rrf


    def get_top_k_hybrid_retrieval(self, query: str, top_k: int=10):

        top_k_by_bm25_ids = [id for id, _ in self.bm25_retriever.search_bm25(query, top_k)]
        den_res = self.dense_retriever.get_top_k_dense_results(query, top_k)
        top_k_by_dense_ids = [id_score_dict["id"] for id_score_dict in den_res]
        return self.rrf.merge_with_rrf([top_k_by_bm25_ids, top_k_by_dense_ids])