import ollama
import httpx

# Qdrant connection
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "institute360"

# Sample documents
documents = [
    "Institute360 provides educational courses for students.",
    "Institute360 offers admission guidance and application support.",
    "Students can explore different academic programs.",
    "Institute360 provides information about educational facilities.",
    "Students can contact the admission department for assistance."
]

points = []

print("Generating embeddings...")

# Generate embeddings
for index, document in enumerate(documents):

    response = ollama.embed(
        model="nomic-embed-text",
        input=document
    )

    embedding = response["embeddings"][0]

    points.append({
        "id": index + 1,
        "vector": embedding,
        "payload": {
            "text": document
        }
    })

    print(f"Embedding generated for document {index + 1}")

# Store embeddings in Qdrant
response = httpx.put(
    f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points?wait=true",
    json={"points": points}
)

if response.status_code == 200:
    print("\nAll embeddings stored successfully!")
    print(response.json())
else:
    print("\nError storing embeddings!")
    print(response.text)
    