from data_loader_component.data_loader import DataLoader
from retrieval.bm25_retriever import BM25Retriever
from pathlib import Path

dat_loader = DataLoader()
bm25_retriever = BM25Retriever(dat_loader)

corpus_parquet_file_path = Path(__file__).parents[1] /  "data" / "corpus.parquet"


bm25_retriever.create_index(corpus_parquet_file_path)