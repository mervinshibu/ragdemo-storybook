# ragdemo-storybook

A simple demonstration of Retrieval-Augmented Generation (RAG) using stories from Grimm's Fairy Tales.

This project prepares a small story dataset and will build a RAG pipeline that answers reading-comprehension questions by retrieving relevant story passages and generating grounded responses with an LLM.

## Dataset

The knowledge base is built from **Grimm's Fairy Tales**, sourced from Project Gutenberg.

Each story is extracted from the HTML version of the book and stored as structured JSON containing:

- story title
- story text

Example structure:

```json
{
  "story_id": 1,
  "title": "THE GOLDEN BIRD",
  "text": "A certain king had a beautiful garden, and in the garden..."
}
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with your OpenAI API key:

```bash
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

## Data Extraction
To extract stories from the HTML source:

```bash
python scripts/extract_stories.py
```

This generates `data/stories.json` which will later be used as the knowledge base for the RAG system.

## Running the Pipeline

After extracting the data, follow these steps to use the RAG system:

1. **Generate Embeddings**
   ```bash
   python src/embeddings.py
   ```
   This reads `data/stories.json`, generates embeddings using the OpenAI API, and saves them to `data/story-vectors.json`.

2. **Initialize ChromaDB**
   ```bash
   python src/init_chroma.py
   ```

3. **Run Console Q&A**
   ```bash
   python src/console_rag_QnA.py
   ```