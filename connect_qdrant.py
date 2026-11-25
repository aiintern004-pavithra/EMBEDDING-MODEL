import httpx

# Qdrant URL
url = "http://localhost:6333"

# Connect to Qdrant
response = httpx.get(url + "/collections")

# Check connection
if response.status_code == 200:
    print("Connected to Qdrant successfully!")

    data = response.json()

    print("Available Collections:")
    print(data)
else:
    print("Connection failed!")
    print(response.text)