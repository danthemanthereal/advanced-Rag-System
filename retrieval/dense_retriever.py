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
            name="vector_db",
            metadata={"hnsw:space": "cosine"}
        )

        collection.add(
            embeddings=embeddings.tolist(),
            documents=doc_texts,
            ids=[
                str(i)
                for i in range(len(doc_texts))
            ],
        )

    def get_top_k_dense_results(self, query: str, top_k: int=10):

        client = chromadb.PersistentClient(path=str(self.vector_db_path))

        collection = client.get_or_create_collection(
            name="vector_db",
            metadata={"hnsw:space": "cosine"}
        )
        embedded_question = self.embedding_model.encode(query, normalize_embeddings=True)
        results = collection.query(
            query_embeddings=[embedded_question.tolist()],
            n_results=5,
            include=[
                "distances"
            ]
        )

        ids = results["ids"][0]
        distances = results["distances"][0]

        top_k_results = []

        for doc_id, distance in zip(ids, distances):
            cosine_similarity = 1 - distance
            top_k_results.append(
                {
                    "id": doc_id,
                    "score": cosine_similarity
                }
            )

        return top_k_results





