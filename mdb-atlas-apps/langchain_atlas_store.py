from dotenv import load_dotenv  
from langchain_voyageai import VoyageAIEmbeddings  
from langchain_mongodb import MongoDBAtlasVectorSearch  
import getpass  
import os  
  
# Load environment variables from .env file  
load_dotenv()  
  
# Choose embedding model and dimensions: 
embeddings = VoyageAIEmbeddings(model="voyage-3-large", output_dimension=2048)  

# load API key:
if not os.environ.get("VOYAGE_API_KEY"):  
    os.environ["VOYAGE_API_KEY"] = getpass.getpass("Enter API key for Voyage AI: ")  
  
# Instantiate the vector store using your MongoDB connection string  
vector_store = MongoDBAtlasVectorSearch.from_connection_string(  
    connection_string=os.getenv("ATLAS_CONNECTION_STRING"),  
    namespace="sample_mflix.embedded_movies",  # This is the namespace where your embeddings are stored
    embedding=embeddings, # This the embedding model/parameters to use
    index_name="vector_index",  
    embedding_key="plot_embedding_voyage_3_large",  # MongoDB field that will contain the embedding for each document.
    text_key="plot"  #  Field that contains the relevant data/text. In this case, the movie's plot description. 
)  
  
# Create a retriever from the vector store  
retriever = vector_store.as_retriever(  
    search_type="similarity", # the default similarity method  
    search_kwargs={"k": 5}  # Return top 5 most similar documents  
)  
  
# Function to test basic similarity retrieval:
def test_retrieval(query):  
    print(f"Input Query: {query}\n\n")  
      
    # Input string, output documents using the defined similarity method and k:    
    docs = retriever.invoke(query)  

    # Iterate through each returned document, printing out the relevant details:   
    for i, doc in enumerate(docs, 1):  
        print(f"Match no. {i}:\n")  
        print(f"Title: {doc.metadata.get('title', 'N/A')}")  
        print(f"Plot: {doc.page_content[:200]}...") 
        print("-" * 30)  
      
    return docs  
  


if __name__ == "__main__":  
    test_queries = [  
        "space adventure with alien encounters",  
        "romantic comedy in New York",  
        "action movie with car chases",  
        "psychological thriller with plot twists"  
    ]  
      
    for query in test_queries:  
        results = test_retrieval(query)  
        print("="*60 + f"\nFound {len(results)} results.\n" + "="*60)  
