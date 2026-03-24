import os
import json
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Ragas 0.2.x imports
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig

load_dotenv()

def get_prechunked_documents(filepath: str) -> list[Document]:
    with open(filepath, 'r', encoding='utf-8') as f:
        stories = json.load(f)
    
    chunks = []
    # We manually chunk the stories by paragraphs so Ragas doesn't have to guess headlines
    for story in stories:
        paragraphs = story['text'].split('\n\n')
        
        current_chunk = ""
        for p in paragraphs:
            current_chunk += p + "\n\n"
            # If our aggregated chunk is over 400 characters, save it and start a new one
            if len(current_chunk) > 400:
                chunks.append(Document(
                    page_content=current_chunk.strip(),
                    metadata={"source": str(story['title']), "story_id": str(story['story_id'])}
                ))
                current_chunk = ""
                
        # Append whatever is left over
        if len(current_chunk.strip()) > 50:
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata={"source": str(story['title']), "story_id": str(story['story_id'])}
            ))
            
    return chunks

def generate_test_set():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'data', 'stories.json')
    output_file = os.path.join(base_dir, 'data', 'eval_dataset.csv')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Loading and chunking documents from {input_file}...")
    chunks = get_prechunked_documents(input_file)
    print(f"Created {len(chunks)} pre-formatted chunks from the stories.")

    print("\nInitializing OpenAI models for Testset Generation...")
    # Ragas 0.2+ requires incredibly strict, deeply nested JSON schema outputs.
    # Use gpt-4o because it supports temperature=0.01 and complex JSON schemas
    llm = ChatOpenAI(model="gpt-4.1-2025-04-14") 

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    try:
        print("\nStarting Ragas generator (Ragas 0.2+ Architecture)...")
        generator = TestsetGenerator.from_langchain(
            llm=llm, 
            embedding_model=embeddings
        )

        print("\nGenerating 20 questions using Ragas 'generate_with_chunks' API...")
        
        # Configure Ragas to retry on network failures and limit concurrent requests preventing server connection drops
        run_config = RunConfig(max_retries=10, max_workers=4, timeout=120)

        # By passing chunks directly to `generate_with_chunks`, Ragas skips the buggy HeadlineSplitter entirely!
        testset = generator.generate_with_chunks(
            chunks, 
            testset_size=20, 
            raise_exceptions=False,
            run_config=run_config
        )
        
        print(f"\nTestset generation complete! Exporting to {output_file}...")
        df = testset.to_pandas()
        df.to_csv(output_file, index=False)
        print("Success! Your test dataset is ready.")
        
    except Exception as e:
        print(f"\nError occurred during generation: {e}")

if __name__ == "__main__":
    generate_test_set()
