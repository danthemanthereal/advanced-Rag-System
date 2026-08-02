import bm25s
from pathlib import Path
from data_loader_component.data_loader import DataLoader


class BM25Retriever:

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.index_dir = (
                Path(__file__).parents[1]
                / "indexes"
                / "bm25"
        )
        self.retriever = bm25s.BM25()


    def create_index(self, data_file_path: Path):
        documents = self.data_loader.get_texts(data_file_path)

        tokens = bm25s.tokenize(
            documents,
            stopwords="en"
        )

        self.retriever.index(tokens)

        self.save_index()

    def save_index(self):
        self.index_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.retriever.save(str(self.index_dir))

    def search_bm25(self, query: str, k: int = 10) -> list[tuple[str, float]]:

        query_tokens = bm25s.tokenize([query], stopwords="en")
        indices, scores = self.retriever.retrieve(query_tokens, k=k)

        doc_ids = self.data_loader.get_documents_ids()

        return [
            (doc_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0].tolist())
        ]