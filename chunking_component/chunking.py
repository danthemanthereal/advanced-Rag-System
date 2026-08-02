from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


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
