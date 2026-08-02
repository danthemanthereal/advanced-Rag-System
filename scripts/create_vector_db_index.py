from data_loader_component.data_loader import DataLoader
from pathlib import Path

from retrieval.dense_retriever import DenseRetriever

embeddings = []

batch_size = 256

data_loader = DataLoader()

doc_text = data_loader.get_texts( Path(__file__).parents[1] / "data" / "corpus.parquet")

dense_retriever = DenseRetriever()

dense_retriever.build_index(doc_text)