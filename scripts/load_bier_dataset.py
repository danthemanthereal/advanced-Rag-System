from datasets import load_dataset
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

corpus = load_dataset("BeIR/fiqa", "corpus", split="corpus")
queries = load_dataset("BeIR/fiqa", "queries", split="queries")
qrels = load_dataset("BeIR/fiqa-qrels", split="test")

corpus.to_parquet(DATA_DIR / "corpus.parquet")
queries.to_parquet(DATA_DIR / "queries.parquet")
qrels.to_parquet(DATA_DIR / "qrels.parquet")

