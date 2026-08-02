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


    def create_index(self, data_file_path: Path):
        documents = self.data_loader.get_texts(data_file_path)

        tokens = bm25s.tokenize(
            documents,
            stopwords="en"
        )

        retriever = bm25s.BM25()
        retriever.index(tokens)

        self.save_index(retriever)

    def save_index(self, retriever):
        self.index_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        retriever.save(str(self.index_dir))