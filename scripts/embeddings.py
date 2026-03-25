import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_and_save_embeddings():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'data', 'stories.json')
    output_file = os.path.join(base_dir, 'data', 'story-vectors.json')

    print(f"Loading stories from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        stories = json.load(f)

    print(f"Found {len(stories)} stories. Generating embeddings...")
    for story in stories:
        print(f"Creating embedding for Story ID {story['story_id']}: {story['title']}")
        text_to_embed = f"Title: {story['title']}\n\nStory: {story['text']}"
        
        response = client.embeddings.create(
            input=text_to_embed,
            model="text-embedding-3-small"
        )
        
        story['embedding'] = response.data[0].embedding

    # Save to new file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stories, f, indent=2)
        
    print(f"\nSuccessfully generated embeddings and saved to {output_file}")

if __name__ == "__main__":
    generate_and_save_embeddings()