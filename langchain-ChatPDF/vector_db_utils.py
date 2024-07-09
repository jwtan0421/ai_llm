from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader

class VectorDBConnector:
    """
    A class to handle the connection and operations related to a vector database using Langchain and OpenAI embeddings.
    """

    def __init__(self, model="text-embedding-ada-002"):
        """
        Initialize the VectorDBConnector with a specified embedding model.

        Args:
            model (str): The name of the embedding model to use. Defaults to "text-embedding-ada-002".
        """
        self.model = model

    def add_file(self, file_path):
        """
        Load a PDF file, split its content into manageable chunks, and store the embeddings in a FAISS vector database.

        Args:
            file_path (str): The path to the PDF file to be processed.
        """
        self.file_path = file_path

        # Load and split the PDF document into pages
        loader = PyMuPDFLoader(self.file_path)
        pages = loader.load_and_split()

        # Split the text content of the pages into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )

        texts = text_splitter.create_documents(
            [page.page_content for page in pages[:4]]
        )

        # Embed the text chunks and store them in a FAISS vector database
        embeddings = OpenAIEmbeddings(model=self.model)
        self.db = FAISS.from_documents(texts, embeddings)

    def get_retriever(self, k=2):
        """
        Get a retriever object to query the vector database for the top-k most similar documents.

        Args:
            k (int): The number of top similar documents to retrieve. Defaults to 2.

        Returns:
            object: A retriever object to perform similarity searches.
        """
        return self.db.as_retriever(search_kwargs={"k": k})
