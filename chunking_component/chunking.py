from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from scripts.create_vector_db_index import embeddings


class ChunkingComponent:

    def __init__(self):
        pass

    def recursive_character_chunking(self, document_content, document_file_name):
        documents = [
            Document(
                page_content=document_content,
                metadata={
                    "source": document_file_name,
                }
            )
        ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        return splitter.split_documents(documents)

    def semantic_chunking(self, document_content):

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        text_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile"
        )

        documents = text_splitter.create_documents(
            [document_content]
        )

        for doc in documents:
            print(doc.page_content)
