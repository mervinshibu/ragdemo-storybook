# Grimm's Archives: Production-Ready RAG Pipeline

A robust demonstration of Retrieval-Augmented Generation (RAG) using stories from Grimm's Fairy Tales.

This project takes raw story text, generates OpenAI embeddings, builds a fast vector database using ChromaDB, and includes a fully functional interactive Q&A console loop. Crucially, it features an automated, synthetic test-set generation script and an objective evaluation pipeline powered by **Ragas**.

## 🏗️ Project Architecture

The codebase strictly follows a clean, modular structure:
* **`src/`**: Contains the live, executable application code.
  * `rag_pipeline.py`: A modular `RAGPipeline` class handling context retrieval and LLM querying.
  * `console_rag_QnA.py`: The live interactive terminal interface.
  * `evaluate_rag.py`: Evaluates the pipeline against the test dataset using Ragas metrics.
* **`scripts/`**: Contains initialization, setup, formatting, and data-gathering workflows.
  * `story_scraper.py`: Extracts raw text from HTML.
  * `embeddings.py`: Generates `text-embedding-3-small` vectors.
  * `init_chroma.py`: Seeds the initial ChromaDB vector store.
  * `generate_testset.py`: Uses OpenAI and Ragas to procedurally generate evaluation questions and ground-truth answers.
  * `word_counter.py`: Utility tool for analyzing corpus sizing.

## 🚀 Setup

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure OpenAI Credentials:**
   Create a `.env` file in the root directory and add your API Key:
   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

## 📚 Building the Knowledge Base

Before you can answer queries, you must build the databases. Run these scripts from any directory:

1. **Extract Stories**
   ```bash
   python scripts/story_scraper.py
   ```
   *Generates `data/stories.json`.*

2. **Generate Embeddings**
   ```bash
   python scripts/embeddings.py
   ```
   *Calls the OpenAI API to embed stories into `data/story-vectors.json`.*

3. **Initialize ChromaDB**
   ```bash
   python scripts/init_chroma.py
   ```
   *Loads your vectors into a local ChromaDB instance at `data/chroma_db`.*

## 🗣️ Running the Application

Once your database is initialized, you can use the RAG pipeline!

```bash
python src/console_rag_QnA.py
```
This drops you into an interactive chat interface. Type an inquiry (e.g. "Tell me a story about a golden bird") to query the Grimm's Archives. Type `exit` to quit.

## 📊 Automated Evaluation (RAGAS)

This project features industry-standard RAG metrics scoring to prove the pipeline's reliability.

1. **Generate the Synthetic Testset**
   ```bash
   python scripts/generate_testset.py
   ```
   *Uses Ragas and your OpenAI `gpt-5` model to autonomously formulate difficult reading-comprehension queries based solely on the extracted stories. Safely bypasses default Ragas text-chunking bugs by pre-formatting the data in Python. Outputs to `data/eval_dataset.csv`.*

2. **Evaluate the Pipeline**
   ```bash
   python src/evaluate_rag.py
   ```
   *Runs the pre-generated test questions directly through the **live pipeline** (`rag_pipeline.py`), judges the output using Ragas LLM-as-a-judge metrics (Faithfulness, LLMContextRecall, FactualCorrectness), and exports the final metrics scores to `data/evaluation_results.csv`.*