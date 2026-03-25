import os
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self):
        # Reliably find the project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        chroma_path = os.path.join(base_dir, "data", "chroma_db")
        
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(name="story_collection")
        self.openai_client = OpenAI()

    def get_answer(self, query: str) -> tuple[str, list[str], list[dict]]:
        """
        Retrieves context and generates an answer for a given query.
        Returns: answer string, list of context strings, and list of metadata dicts.
        """
        # 1. Retrieve the most relevant chunks using embeddings
        emb_res = self.openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = emb_res.data[0].embedding
        
        db_res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas"]
        )
        
        contexts = db_res['documents'][0] if db_res['documents'] else []
        metadatas = db_res['metadatas'][0] if db_res['metadatas'] else []
        
        # 2. Answer the question using the retrieved context
        input_text = "Context: " + "\n".join(contexts) + "\n\nQuestion: " + query
        
        try:
            response = self.openai_client.responses.create(
                model="gpt-5-nano-2025-08-07",
                reasoning={"effort": "low"},
                instructions="You are a question-answering assistant. Use ONLY the provided context to answer the question. If the answer is not explicitly stated in the context, say \"I don't know\".Do NOT make up information. Do NOT use prior knowledge. Keep the answer concise and grounded in the context.",
                input=input_text,
            )
            answer = response.output_text
        except Exception:
            # Fallback to standard OpenAI completions if the experimental endpoint fails
            chat_res = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a question-answering assistant. Use ONLY the provided context. If the answer is not explicitly stated, say 'I don't know'. Do NOT make up information."},
                    {"role": "user", "content": input_text}
                ]
            )
            answer = chat_res.choices[0].message.content

        return answer, contexts, metadatas
