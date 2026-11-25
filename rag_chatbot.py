
import ollama
import httpx

# Qdrant connection
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "institute360"

# Models
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"


# Step 1: Get the user's question
question = input("You: ").strip()

if not question:
    print("Please enter a question.")
    exit()


# Step 2: Convert the question into an embedding
print("\nSearching for relevant information...")

try:
    embedding_response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=question
    )

    query_vector = embedding_response["embeddings"][0]

except Exception as e:
    print("\nError generating embedding:", e)
    exit()


# Step 3: Search for relevant information in Qdrant
try:
    search_response = httpx.post(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/query",
        json={
            "query": query_vector,
            "limit": 3,
            "with_payload": True,
            "with_vector": False
        },
        timeout=30.0
    )

    search_response.raise_for_status()

    results = search_response.json()

except Exception as e:
    print("\nError connecting to Qdrant:", e)
    exit()


# Step 4: Extract the retrieved information
print("\n========== RETRIEVED INFORMATION ==========")

relevant_information = []

points = results.get("result", {}).get("points", [])

if not points:
    print("No relevant information found in Qdrant.")

for point in points:

    score = point.get("score", 0)

    text = point.get("payload", {}).get("text", "")

    print("\nSimilarity Score:", score)
    print("Retrieved Text:", text)

    if text:
        relevant_information.append(text)


# Step 5: Prepare the context
context = "\n".join(relevant_information)

print("\n========== FINAL CONTEXT ==========")
print(context if context else "No context available.")

# Stop if no information is retrieved
if not context.strip():
    print("\nInstitute360 Chatbot:")
    print(
        "I couldn't find relevant information in my knowledge base. "
        "Please try another question."
    )
    exit()


# Step 6: Generate an answer using Llama 3.2
print("\nGenerating answer...")

try:
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Institute360 assistant. "
                    "Answer the user's question using only the "
                    "provided context. "
                    "If the context does not contain the answer, "
                    "clearly say that you do not have enough "
                    "verified Institute360 information. "
                    "Do not invent facts."
                )
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}

Provide a clear, accurate and helpful answer.
"""
            }
        ]
    )

except Exception as e:
    print("\nError generating chatbot response:", e)
    exit()


# Step 7: Display the answer
print("\n========== INSTITUTE360 CHATBOT ==========")
print(response["message"]["content"])
print("\n==========================================")
