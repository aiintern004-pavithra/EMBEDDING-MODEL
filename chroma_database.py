
import ollama
import chromadb

# Step 1: Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Step 2: Create a collection
collection = client.get_or_create_collection(
    name="institute360_documents"
)

# Step 3: Institute360 documents
documents = [
    "Institute360 provides courses for students.",
    "Institute360 offers admission facilities for various courses.",
    "Students can learn Python programming at Institute360.",
    "Institute360 provides information about course fees.",
    "Students can contact Institute360 for admission enquiries.",
    "Institute360 offers technical and professional training programs."
]

# Step 4: Generate embeddings and store documents
print("Generating embeddings and storing documents...")

for i, document in enumerate(documents):

    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=document
    )

    embedding = response["embedding"]

    collection.upsert(
        ids=[str(i)],
        documents=[document],
        embeddings=[embedding]
    )

print("Documents successfully stored in ChromaDB!")

# Step 5: Semantic search
while True:

    question = input("\nEnter your question (type exit to stop): ")

    if question.lower() == "exit":
        print("Exiting semantic search...")
        break

    # Generate embedding for the question
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=question
    )

    question_embedding = response["embedding"]

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    print("\nRelevant Documents:")

    for document in results["documents"][0]:
        print("-", document)
