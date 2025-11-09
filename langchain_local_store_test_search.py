from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

import getpass
import os


# Load environment variables from .env file
load_dotenv()

# Choose embedding model and load API key: 
embeddings = VoyageAIEmbeddings(model="voyage-3-lite")

if not os.environ.get("VOYAGE_API_KEY"):
    os.environ["VOYAGE_API_KEY"] = getpass.getpass("Enter API key for Voyage AI: ")

# Instantiate the in-memory vectore store with defined embedding function:
vector_store = InMemoryVectorStore(embeddings) 

# Load pdf file locally: 
file_path = "./documents/mongodb_atlas_whitepaper.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

# Use langchain's text splitter: 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

# Split the loaded PDF into chunks:
all_splits = text_splitter.split_documents(docs)

# Create test embeddings with the first two chunks:
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
# print(f"Generated vectors of length {len(vector_1)}\n")
# print(vector_1[:10])

# index the documents/chunks within the vector store: 
""" 
After indexing, the documents will be queryable. Some functions to try:
- similarity_search()
- similarity_search_with_score()
- embed_query()
"""
ids = vector_store.add_documents(documents=all_splits)

# Run a test similarity search: 
results = vector_store.similarity_search(
    "Data privacy"
)


output_string = " ".join([doc.page_content for doc in results])  
print(output_string)  


