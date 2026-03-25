from rag_pipeline import RAGPipeline

def main():
    print("Initializing pipeline...")
    pipeline = RAGPipeline()
    
    print("\nWelcome to the Grimm's Archives!")
    while True:
        query = input("\nEnter your query or 'exit' to exit: ")
        if query.lower() == 'exit':
            break
            
        answer, _, metadatas = pipeline.get_answer(query)
        
        print("\n--- Sources ---")
        for metadata in metadatas:
            print(f"Story ID: {metadata.get('story_id')} | Title: {metadata.get('title')}")
            
        print("\n--- Answer ---")
        print(answer)

if __name__ == "__main__":
    main()
