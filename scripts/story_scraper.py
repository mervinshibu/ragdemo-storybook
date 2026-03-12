from bs4 import BeautifulSoup
import json
import re
from pathlib import Path

html_path = Path("../data/grimm-fairy-tales.html")
output_path = Path("../data/stories.json")

with html_path.open("r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

chapters = soup.select("div.chapter")
stories = []

story_id = 1
for chapter in chapters:
    # Find title
    h2 = chapter.find("h2")
    if not h2:
        continue

    title = h2.get_text(" ", strip=True)

    # Re-parse chapter HTML so we can safely remove title
    chapter_copy = BeautifulSoup(str(chapter), "html.parser")
    h2_copy = chapter_copy.find("h2")
    if h2_copy:
        h2_copy.decompose()

    # Extract all remaining text
    text = chapter_copy.get_text("\n", strip=True)

    # Normalize whitespace but keep paragraph-ish breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text).strip()

    if not text:
        continue

    stories.append({
        "story_id": story_id,
        "title": title,
        "text": text,
    })
    story_id += 1

with output_path.open("w", encoding="utf-8") as f:
    json.dump(stories, f, ensure_ascii=False, indent=2)

print(f"Extracted {len(stories)} stories to {output_path}")