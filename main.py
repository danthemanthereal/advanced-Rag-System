from collections import defaultdict
from data_loader_component.data_loader import DataLoader
from pathlib import Path
from tqdm import tqdm

from reciprocal_rank_fusion.rrf import ReciprocalRankFusion
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from scripts.create_vector_db_index import dense_retriever

"""SEED = 42
RERANK_SAMPLE_SIZE=10

DATA_DIR = Path(__file__).resolve().parents[0] / "data"

data_loader = DataLoader()

queries = data_loader.get_data_as_pd(DATA_DIR / "queries.parquet")
qrels_df = data_loader.get_data_as_pd(DATA_DIR / "qrels.parquet")

qrels: dict[str, dict[str, int]] = defaultdict(dict)
for _, row in qrels_df.iterrows():
    qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])

queries_with_qrels = queries[queries["_id"].astype(str).isin(qrels.keys())].copy()
sample = queries_with_qrels.sample(n=RERANK_SAMPLE_SIZE, random_state=SEED)

results: dict[str, list[float]] = defaultdict(list)
for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Evaluating"):
    query_id = str(row["_id"])
    query_text = row["text"]
    relevant = qrels[query_id]"""

data_loader = DataLoader()
dense_retriever = DenseRetriever()
bm25_retriever = BM25Retriever(data_loader=data_loader)
rrf = ReciprocalRankFusion(60)

hybrid_retriever = HybridRetriever(bm25_retriever=bm25_retriever,
                                   dense_retriever=dense_retriever,
                                   rrf=rrf)







