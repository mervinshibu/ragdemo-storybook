import json
import os

def main():
    # Get the base directory reliably (storybook-rag)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, 'data', 'stories.json')
        
    with open(json_path, 'r', encoding='utf-8') as file:
        stories = json.load(file)
        
    total_words = 0
    
    for story in stories:
        title = story.get('title', '')
        text = story.get('text', '')
        
        # Use the concatenated string of title + text
        concatenated_string = f"{title} {text}"
        
        # Word count is the length of the string split by whitespace
        word_count = len(concatenated_string.split())
        total_words += word_count
        
        print(f"Story ID {story.get('story_id')}: '{title}' - {word_count} words")
        
    print("-" * 40)
    print(f"Total word count for all stories combined: {total_words}")

if __name__ == "__main__":
    main()
