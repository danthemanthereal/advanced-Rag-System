from pathlib import Path

from sentence_transformers import CrossEncoder
from data_loader_component.data_loader import DataLoader


class Reranker:

    def __init__(self, model_name: str,
                 data_loader: DataLoader,
                 ):
        self.cross_encoder_model = CrossEncoder(
            model_name=model_name,
        )
        self.data_loader = data_loader

    def rerank(self, query: str, document_ids: list[str], top_k: int = 5):
        document_text = self.data_loader.get_documents_by_id(
            Path(__file__).parents[1] / "data" / "corpus.parquet",
            document_ids
        )

        pairs = [
            [query, doc]
            for doc in document_text
        ]

        scores = self.cross_encoder_model.predict(pairs)

        return [
            doc
            for _, doc in sorted(
            zip(scores, document_text),
            reverse=True
        )[:top_k]
        ]






