from pathlib import Path

import pandas as pd

class DataLoader:

    def __init__(self, ):
        pass


    def get_data_as_pd(self, data_file_path: Path) -> pd.DataFrame:
        return pd.read_parquet(data_file_path)

    def get_texts(self, data_file_path: Path) -> list[str]:
        corpus = self.get_data_as_pd(data_file_path)
        return corpus["text"].tolist()[:50]

    def get_documents_ids(self, data_file_path: Path = Path(__file__).parents[1] / "data" / "corpus.parquet") -> list[str]:
        documents = self.get_data_as_pd(data_file_path)
        return documents["_id"].tolist()