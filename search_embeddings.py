import ollama
import httpx

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "institute360"

# Ask a question
question = "How can I get help with joining the institute?"

print("User Question:", question)

# Convert question into an embedding
response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

query_vector = response["embeddings"][0]

print("\nSearching Qdrant...")

# Search for similar vectors
search_response = httpx.post(
    f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/query",
    json={
        "query": query_vector,
        "limit": 3,
        "with_payload": True
    }
)

results = search_response.json()

print("\nRelevant Information:")

for point in results["result"]["points"]:
    print("\nText:", point["payload"]["text"])
    print("Similarity Score:", point["score"])