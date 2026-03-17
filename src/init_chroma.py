import chromadb
import json
import uuid

chroma_client = chromadb.PersistentClient(path="data/chroma_db")

collection = chroma_client.get_or_create_collection(name="story_collection")
embeddings = []
documents = []
metadatas = []
ids = []

with open("data/story-vectors.json", "r") as f:
    data = json.load(f)
    ids = [str(uuid.uuid4()) for _ in data]
    embeddings = [item["embedding"] for item in data]
    documents  = [f"{item['title']}\n\n{item['text']}" for item in data]
    metadatas = [{"story_id": item["story_id"], "title": item["title"]} for item in data]


collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas,
)