from openai import OpenAI
from dotenv import load_dotenv
import chromadb

load_dotenv()

chroma_client = chromadb.PersistentClient(path="data/chroma_db")
openai_client = OpenAI()

collection = chroma_client.get_or_create_collection(name="story_collection")

while True:
    query = input("Enter your query or 'exit' to exit: ")
    if query.lower() == 'exit':
        break
    response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["metadatas", "distances", "documents"]
    )
    metadatas = result['metadatas'][0]
    for metadata in metadatas:
        print(f"Story ID: {metadata['story_id']}")
        print(f"Title: {metadata['title']}")
    inputText = "Context: " + "\n".join([doc for doc in result['documents'][0]]) + "\n\nQuestion: " + query
    response = openai_client.responses.create(
        model="gpt-5-nano-2025-08-07",
        reasoning={"effort": "low"},
        instructions="You are a question-answering assistant. Use ONLY the provided context to answer the question. If the answer is not explicitly stated in the context, say \"I don't know\".Do NOT make up information. Do NOT use prior knowledge. Keep the answer concise and grounded in the context.",
        input=inputText,
    )
    print("Answer: " + response.output_text)

