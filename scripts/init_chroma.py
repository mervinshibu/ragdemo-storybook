import os
import chromadb
import json
import uuid

# Ensure robust absolute pathing regardless of where the script is executed from
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(base_dir, "data", "chroma_db")
vectors_file = os.path.join(base_dir, "data", "story-vectors.json")

chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection(name="story_collection")

embeddings = []
documents = []
metadatas = []
ids = []

print(f"Loading vectors from {vectors_file}...")
with open(vectors_file, "r") as f:
    data = json.load(f)
    ids = [str(uuid.uuid4()) for _ in data]
    embeddings = [item["embedding"] for item in data]
    documents  = [f"{item['title']}\n\n{item['text']}" for item in data]
    metadatas = [{"story_id": str(item["story_id"]), "title": str(item["title"])} for item in data]

print(f"Loading {len(ids)} documents into ChromaDB at {db_path}...")
collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas,
)
print("Database initialization complete!")
