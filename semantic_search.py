
import ollama
import math

# Step 1: Sample Institute360 documents
documents = [
    "Institute360 provides courses and admission facilities.",
    "Students can apply for admission through the institute.",
    "The institute offers various educational courses.",
    "Students can contact the institute for fee details.",
    "The institute provides information about examinations."
]

# Step 2: Generate embeddings
def generate_embedding(text):
    response = ollama.embed(
        model="nomic-embed-text",
        input=text
    )
    return response["embeddings"][0]


# Step 3: Calculate cosine similarity
def cosine_similarity(vector1, vector2):

    if len(vector1) != len(vector2):
        raise ValueError("Embedding dimensions do not match.")

    dot_product = sum(
        a * b for a, b in zip(vector1, vector2)
    )

    magnitude1 = math.sqrt(sum(a * a for a in vector1))
    magnitude2 = math.sqrt(sum(b * b for b in vector2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


# Step 4: Generate document embeddings
print("=" * 55)
print("       INSTITUTE360 SEMANTIC SEARCH")
print("=" * 55)

print("\nGenerating document embeddings...")

document_embeddings = []

for document in documents:
    embedding = generate_embedding(document)
    document_embeddings.append(embedding)

print("Document embeddings generated successfully!")

print("Total Documents:", len(documents))
print("Vector Dimensions:", len(document_embeddings[0]))


# Step 5: Get user question
query = input("\nEnter your question: ").strip()

if not query:
    print("Please enter a valid question.")
    raise SystemExit

# Step 6: Generate query embedding
query_embedding = generate_embedding(query)


# Step 7: Calculate similarity scores
results = []

for document, doc_embedding in zip(
    documents, document_embeddings
):

    similarity = cosine_similarity(
        query_embedding,
        doc_embedding
    )

    results.append((document, similarity))


# Step 8: Sort by similarity
results.sort(
    key=lambda item: item[1],
    reverse=True
)


# Step 9: Display search results
print("\n" + "=" * 55)
print("             SEMANTIC SEARCH RESULTS")
print("=" * 55)

for rank, (document, score) in enumerate(results, start=1):

    percentage = score * 100

    print(f"\nRank: {rank}")
    print("Document:", document)
    print("Similarity Score:", round(score, 4))
    print("Similarity Percentage:", round(percentage, 2), "%")

    if score >= 0.70:
        print("Relevance: High")
    elif score >= 0.50:
        print("Relevance: Medium")
    else:
        print("Relevance: Low")


# Step 10: Display best match
best_document, best_score = results[0]

print("\n" + "=" * 55)
print("                BEST MATCH")
print("=" * 55)

if best_score >= 0.50:
    print("Most Relevant Document:", best_document)
    print("Best Similarity Score:", round(best_score, 4))
else:
    print("No sufficiently relevant document found.")

print("\nSemantic Search Completed Successfully!")
print("=" * 55)
