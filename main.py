from collections import defaultdict
from data_loader_component.data_loader import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from evaluator_component.evaluator import Evaluator
from reciprocal_rank_fusion.rrf import ReciprocalRankFusion
from reranker_component.reranker import Reranker
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever

DATA_DIR = Path(__file__).resolve().parents[0] / "data"

data_loader = DataLoader()
dense_retriever = DenseRetriever()
bm25_retriever = BM25Retriever(data_loader=data_loader)
rrf = ReciprocalRankFusion(60)
hybrid_retriever = HybridRetriever(bm25_retriever=bm25_retriever,
                                   dense_retriever=dense_retriever,
                                   rrf=rrf)
reranker = Reranker(
    data_loader=data_loader,
    model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)

evaluator = Evaluator()

queries = data_loader.get_data_as_pd(DATA_DIR / "queries.parquet")
qrels_df = data_loader.get_data_as_pd(DATA_DIR / "qrels.parquet")

qrels: dict[str, dict[str, int]] = defaultdict(dict)
for _, row in qrels_df.iterrows():
    qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])

queries_with_qrels = queries[queries["_id"].astype(str).isin(qrels.keys())].copy()

sample = queries_with_qrels.head(50)

results = []
for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Evaluating"):
    query_id = str(row["_id"])
    query_text = row["text"]
    relevant = qrels[query_id]

    hybrid_rev = hybrid_retriever.get_top_k_hybrid_retrieval(query_text, 20)
    ids = [str(id) for id, _ in hybrid_rev]
    res = reranker.rerank(query_text, ids, 5)

    current_result = evaluator.ndcg_at_k(ids, relevant,5)
    print("current result:", current_result )
    results.append(current_result)


print(f"Mean NDCG@5: {np.mean(results) * 100:.2f}")










