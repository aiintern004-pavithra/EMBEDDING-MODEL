
import ollama
import time

# Step 1: Sample text
text = "Institute360 provides courses and admission facilities."

print("=" * 55)
print("       INSTITUTE360 - PYTHON EMBEDDING MODEL")
print("=" * 55)

start_time = time.time()

# Step 2: Generate embedding
print("\nGenerating embeddings...")

response = ollama.embed(
    model="nomic-embed-text",
    input=text
)

# Step 3: Extract vector
embedding = response["embeddings"][0]

# Step 4: Display original text
print("\nOriginal Text:")
print(text)

# Step 5: Display vector dimensions
print("\nVector Dimensions:", len(embedding))

# Step 6: Display first 10 vector values
print("\nFirst 10 Vector Values:")
print(embedding[:10])

# Step 7: Display vector statistics
print("\nVector Statistics:")
print("Minimum Value:", min(embedding))
print("Maximum Value:", max(embedding))

# Step 8: Execution time
end_time = time.time()

print("\nExecution Time:", round(end_time - start_time, 3), "seconds")

print("\nEmbedding generated successfully!")
print("=" * 55)
