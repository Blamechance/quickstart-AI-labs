from dotenv import load_dotenv  
import pymongo  
import os  
import getpass  
import voyageai  
  
# Load environment variables from .env file  
load_dotenv()  
  
# Get MongoDB connection string  
if not os.environ.get("ATLAS_CONNECTION_STRING"):  
    os.environ["ATLAS_CONNECTION_STRING"] = getpass.getpass("Enter MongoDB connection string: ")  
  
# Get Voyage AI API key  
if not os.environ.get("VOYAGE_API_KEY"):  
    os.environ["VOYAGE_API_KEY"] = getpass.getpass("Enter Voyage AI API key: ")  
  
# Connect to MongoDB Atlas  
client = pymongo.MongoClient(os.getenv("ATLAS_CONNECTION_STRING"))  
  
# Initialize Voyage AI client  
vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))  
  
def generate_embedding(text):  
    """  
    Generate embedding vector from text using Voyage AI.  
    """  
    result = vo.embed(  
        texts=[text],  
        model="voyage-3-large",  
        input_type="query",
        output_dimension=2048  
    )  
    return result.embeddings[0]  
  
# Get user input  
user_query = input("Enter your search query: ")  
  
print(f"\n🔍 Searching for: '{user_query}'")  
print("⏳ Generating Query embedding...")  
  
# Generate embedding from user query  
query_vector = generate_embedding(user_query)  
  
print(f"✅ Query Embedding generated ({len(query_vector)} dimensions)")  
  
# Define pipeline  
pipeline = [  
    {  
        '$vectorSearch': {  
            'index': 'vector_index',  
            'path': 'plot_embedding_voyage_3_large',  
            'queryVector': query_vector,  
            'numCandidates': 150,  
            'limit': 10  
        }  
    },  
    {  
        '$project': {  
            '_id': 0,  
            'plot': 1,  
            'title': 1,  
            'score': {'$meta': 'vectorSearchScore'}  
        }  
    }  
]  
  
# Run search  
print("🔎 Searching Atlas Cluster ...\n")  
results = client["sample_mflix"]["embedded_movies"].aggregate(pipeline)  
  
# Display results  
print("=" * 80)  
print(f"RESULTS FOR: '{user_query}'")  
print("=" * 80)  
  
for i, doc in enumerate(results, 1):  
    print(f"\n{i}. {doc['title']}")  
    print(f"   Score: {doc['score']:.4f}")  
    print(f"   Plot: {doc['plot'][:400]}...")  
    print("-" * 80)  
  
print("\n✨ Search complete!")  
