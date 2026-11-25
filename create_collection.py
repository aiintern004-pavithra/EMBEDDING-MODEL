import httpx

url = "http://localhost:6333"

# Collection details
collection_name = "institute360"

# Create collection
response = httpx.put(
    f"{url}/collections/{collection_name}",
    json={
        "vectors": {
            "size": 768,
            "distance": "Cosine"
        }
    }
)

print("Collection Creation Response:")
print(response.json())