from sentence_transformers import SentenceTransformer

print("Downloading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model downloaded successfully!")
print("Vector dimensions:", model.get_sentence_embedding_dimension())