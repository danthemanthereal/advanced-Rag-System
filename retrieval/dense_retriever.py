from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np




class DenseRetriever:

    def __init__(self,):
        self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-small')
        self.batch_size = 16
        self.vector_db_path = Path(__file__).parents[1] / "indexes" / "dense"

    def build_index(self,doc_texts: list[str]):

        embeddings = []

        for i in tqdm(
                range(0, len(doc_texts), self.batch_size),
                desc="Creating embeddings"
        ):
            batch = doc_texts[i:i + self.batch_size]

            batch_embeddings = self.embedding_model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            embeddings.extend(batch_embeddings)

        embeddings = np.array(
            embeddings,
            dtype=np.float32
        )

        client = chromadb.PersistentClient(
            path=str(self.vector_db_path)
        )

        collection = client.get_or_create_collection(
            name="vector_db"
        )

        collection.add(
            embeddings=embeddings.tolist(),
            documents=doc_texts,
            ids=[
                str(i)
                for i in range(len(doc_texts))
            ],
        )






