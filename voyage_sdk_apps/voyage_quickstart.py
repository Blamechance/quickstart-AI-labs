import voyageai
import numpy as np
import os
import getpass

from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.neighbors import NearestNeighbors  

# Load environment variables from .env file
load_dotenv()

vo = voyageai.Client() # Use voyageai.AsyncClient() for non-blocking api requests. 

result = vo.embed(["hello world"], model="voyage-3.5")

# Corpus:
documents = [
    "The Mediterranean diet emphasizes fish, olive oil, and vegetables, believed to reduce chronic diseases.", 
    "Photosynthesis in plants converts light energy into glucose and produces essential oxygen.",
    "20th-century innovations, from radios to smartphones, centered on electronic advancements.",
    "Rivers provide water, irrigation, and habitat for aquatic species, vital for ecosystems.",
    "Apple’s conference call to discuss fourth fiscal quarter results and business updates is scheduled for Thursday, November 2, 2023 at 2:00 p.m. PT / 5:00 p.m. ET.",
    "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' endure in literature.",
    "The beloved Granny Smith Festival is set to take place on Saturday 18 October 2025, from 9:00am to 8:30pm in Eastwood, NSW. ",
    "Companies hold earnings conference calls on the same day or the day after they release their quarterly reports, which are subject to SEC filing requirements. The exact time can vary, but they are often scheduled during peak business hours, though a study noted that late afternoon calls were sometimes more negative. ",
    "Local Farmer's Market Apple Festival planning committee call scheduled for this Thursday at 2:30 p.m. to finalize event details."
]

# Search Parameters:
k_size = 3
plaintext_query = "When is Apple's conference call scheduled?"


def inspect_tokens(documents): 
    # length of input list: 
    print(f"Input list count: {len(documents)}")

    # Test tokenize the corpus:
    tokenized = vo.tokenize(documents, model="voyage-3.5")
    for i in range(len(documents)):
        print(f"\nDocument No.{i}:")
        print(tokenized[i].tokens)

    # length of output list: 
    print(f"\nOutput tokenized list count: {len(tokenized)}\n")

    # Manual count of tokens: 
    total_tokens = sum(len(doc) for doc in tokenized)  
    print(f"Manual count of tokens: {total_tokens}")  

    # Uinge count_tokens instead:
    total_tokens = vo.count_tokens(documents, model="voyage-3.5")
    print(f"count_tokens: {total_tokens}")

def cosine_search(query_embedding, document_embeddings, k=k_size):
    """ 
    Performs a k-nearest neighbor search using cosine similarity. Finds the k most similar document embeddings to the query embedding.  
    
    Return: k most relevant matched documents, as:
        1. A list of embeddings. 
        2. A list of indexes, for the matched documents as per the corpus list. 
    """  
    # Calculate how similar the query is to each document  
    similarities = cosine_similarity([query_embedding], document_embeddings)[0] # this returns a ndarray of essentially similarity scores 
      
    # Narrow it down based on size k:   
    top_k_indices = np.argsort(similarities)[::-1][:k]  

    # Get the corresponding similarity scores  
    top_k_scores = similarities[top_k_indices]  
      
    return document_embeddings[top_k_indices], top_k_indices, top_k_scores

# Embed the documents
documents_embeddings = np.array(vo.embed(
    documents, model="voyage-3.5", input_type="document"
).embeddings)

# Embed query string for searching:
query_embedding = vo.embed([plaintext_query], model="voyage-3.5", input_type="query").embeddings[0]


# Cosine Results:
cosine_embeddings_result, cosine_indices, cosine_score = cosine_search(query_embedding, documents_embeddings, k_size)  

print("\n\n=== Cosine Search Result ===")  
for i, index in enumerate(cosine_indices):  
    print(f"{i+1}. Document {index+1}:")  
    print(f"   {documents[index]}") 
    print(f"Score: {cosine_score[i]}\n")

# Reranking: 
print("\n\n=== Re-ranked Documents Search Result ===")  

documents_reranked = vo.rerank(plaintext_query, documents, model="rerank-2.5", top_k=k_size)

for r in documents_reranked.results:
    print(f"Document no. {r.index + 1}: {r.document}")
    print(f"Index: {r.index}")
    print(f"Relevance Score: {r.relevance_score}\n")


