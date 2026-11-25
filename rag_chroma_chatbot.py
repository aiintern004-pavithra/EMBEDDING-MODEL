
import ollama
import chromadb

# Step 1: Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(
    name="institute360_documents"
)

print("Institute360 Hybrid RAG Chatbot Started!")
print("Ask any question. Type 'exit' to stop.")

while True:

    # Step 2: Get user question
    question = input("\nYou: ").strip()

    if not question:
        continue

    if question.lower() == "exit":
        print("Chatbot stopped!")
        break

    try:

        # Step 3: Classify the question
        classification = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the question into exactly one category: "
                        "INSTITUTE or GENERAL.\n"
                        "INSTITUTE: Questions about Institute360, its "
                        "courses, admissions, fees, facilities, training, "
                        "services, staff, or policies.\n"
                        "GENERAL: All other questions, including "
                        "programming, AI, machine learning, science, "
                        "technology, and general knowledge.\n"
                        "Reply with only INSTITUTE or GENERAL."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        category = (
            classification["message"]["content"]
            .strip()
            .upper()
        )

        print("\nQuestion Category:", category)

        # Step 4: Handle Institute360 questions
        if category == "INSTITUTE":

            print("Searching Institute360 knowledge base...")

            # Generate question embedding
            embedding_response = ollama.embeddings(
                model="nomic-embed-text",
                prompt=question
            )

            question_embedding = embedding_response["embedding"]

            # Search ChromaDB
            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=3
            )

            retrieved_documents = results["documents"][0]

            # Check whether documents were retrieved
            if not retrieved_documents:
                print(
                    "\nInstitute360 Hybrid Bot: "
                    "I don't have relevant information "
                    "about this topic in my Institute360 knowledge base."
                )
                continue

            # Prepare context
            context = "\n\n".join(retrieved_documents)

            # Generate answer using retrieved context
            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the official Institute360 "
                            "information assistant.\n"
                            "Answer questions about Institute360 "
                            "using ONLY the provided context.\n"
                            "Do not use your general knowledge to "
                            "fill missing Institute360 information.\n"
                            "Do not invent fees, admission details, "
                            "course information, or policies.\n"
                            "If the context does not contain the "
                            "answer, clearly say:\n"
                            "'I don't have that information in my "
                            "Institute360 knowledge base.'\n"
                            "Provide clear and simple answers."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Retrieved Context:\n{context}\n\n"
                            f"User Question: {question}"
                        )
                    }
                ]
            )

        # Step 5: Handle general questions
        elif category == "GENERAL":

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful general AI assistant.\n"
                            "Answer educational, technical, "
                            "programming, science, and general "
                            "knowledge questions.\n"
                            "Explain concepts clearly and use "
                            "examples when useful."
                        )
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            )

        else:
            print(
                "\nInstitute360 Hybrid Bot: "
                "Unable to classify the question. Please try again."
            )
            continue

        # Step 6: Display answer
        print(
            "\nInstitute360 Hybrid Bot:",
            response["message"]["content"]
        )

    except Exception as e:

        print("\nError:", str(e))
        print("Please check whether Ollama and ChromaDB are running.")
