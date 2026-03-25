import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Ragas evaluators
from ragas import evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_openai import ChatOpenAI

from rag_pipeline import RAGPipeline

load_dotenv()

def evaluate_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    testset_path = os.path.join(base_dir, 'data', 'eval_dataset.csv')
    output_path = os.path.join(base_dir, 'data', 'evaluation_results.csv')

    print(f"Loading test dataset from {testset_path}...")
    df = pd.read_csv(testset_path)
    
    if 'user_input' not in df.columns or 'reference' not in df.columns:
        raise ValueError("Missing 'user_input' or 'reference' columns in the generated dataset.")

    print("Initializing Modular RAG Pipeline...")
    pipeline = RAGPipeline()
    results_data = []

    print(f"\nRunning {len(df)} questions through your pipeline...")
    for index, row in df.iterrows():
        question = row['user_input']
        reference = row['reference']
        print(f"[{index+1}/{len(df)}] Querying DB: {question[:50]}...")
        
        # Use our centralized logic!
        answer, contexts, _ = pipeline.get_answer(question)

        results_data.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": reference
        })

    eval_dataset = Dataset.from_pandas(pd.DataFrame(results_data))

    print("\nPassing pipeline results to RAGAS for scoring...")
    evaluator_llm = ChatOpenAI(model="gpt-5-2025-08-07")
    
    metrics = [
        LLMContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        FactualCorrectness(llm=evaluator_llm)
    ]
    
    score = evaluate(eval_dataset, metrics=metrics)
    
    print("\n========= RAGAS EVALUATION RESULTS =========")
    print(score)
    print("=============================================\n")

    results_df = score.to_pandas()
    results_df.to_csv(output_path, index=False)
    print(f"Detailed metric scores saved to {output_path}")

if __name__ == "__main__":
    evaluate_pipeline()
