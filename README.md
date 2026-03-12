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
  "title": "The Frog-King, or Iron Henry",
  "text": "In olden times when wishing still helped one..."
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

## Data Extraction
To extract stories from the HTML source:

```bash
python scripts/extract_stories.py
```

This generates `data/stories.json` which will later be used as the knowledge base for the RAG system.