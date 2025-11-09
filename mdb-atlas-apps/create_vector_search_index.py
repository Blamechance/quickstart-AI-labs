from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel
import time
import getpass
import os

# Load environment variables from .env file
load_dotenv()

if not os.environ.get("ATLAS_CONNECTION_STRING"):
    os.environ["ATLAS_CONNECTION_STRING"] = getpass.getpass("Enter connection string: ")

# Connect to your Atlas deployment
uri = os.getenv("ATLAS_CONNECTION_STRING")
client = MongoClient(uri)

# Access your database and collection
database = client["sample_mflix"]
collection = database["embedded_movies"]

# Create your index model, then create the search index
search_index_model = SearchIndexModel(
  definition={
    "fields": [
      {
        "type": "vector",
        "path": "plot_embedding_voyage_3_large",
        "numDimensions": 2048,
        "similarity": "dotProduct",
        "quantization": "scalar"
      }
    ]
  },
  name="vector_index",
  type="vectorSearch"
)

result = collection.create_search_index(model=search_index_model)
print("New search index named " + result + " is building.")

# Wait for initial sync to complete
print("Polling to check if the index is ready. This may take up to a minute.")
predicate=None
if predicate is None:
  predicate = lambda index: index.get("queryable") is True

while True:
  indices = list(collection.list_search_indexes(result))
  if len(indices) and predicate(indices[0]):
    break
  time.sleep(5)
print(result + " is ready for querying.")

client.close()
