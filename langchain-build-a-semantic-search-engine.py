from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
import getpass
import os


# Load environment variables from .env file
load_dotenv()

# Choose embedding model and load API key: 
embeddings = VoyageAIEmbeddings(model="voyage-3-lite")

if not os.environ.get("VOYAGE_API_KEY"):
    os.environ["VOYAGE_API_KEY"] = getpass.getpass("Enter API key for Voyage AI: ")


# Load pdf file locally: 
file_path = "./documents/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

# Create the text splitter class: 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

# Split the loaded PDF into chunks:
all_splits = text_splitter.split_documents(docs)

# Create test embeddings with the first two chunks:
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)


assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])