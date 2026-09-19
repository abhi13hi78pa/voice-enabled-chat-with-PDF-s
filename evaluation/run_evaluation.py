import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chain import ask_question

def load_dataset():
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.json')
    if not os.path.exists(dataset_path):
        return []
    with open(dataset_path, 'r') as f:
        return json.load(f)

def run_eval():
    dataset = load_dataset()
    if not dataset:
        print("No evaluation dataset found.")
        return

    print(f"Running evaluation on {len(dataset)} examples...")
    
    # In a full project, we would use RAGAS or LangChain evaluators.
    # Here we simulate the evaluation based on basic substring checks or LLM-as-a-judge concepts.
    
    for i, item in enumerate(dataset):
        question = item['question']
        expected_answer = item['expected_answer']
        
        print(f"\n--- Example {i+1} ---")
        print(f"Q: {question}")
        
        try:
            # Assume document is already indexed and we don't filter by doc_id for eval
            answer, docs = ask_question(question, chat_history=[])
            print(f"Generated A: {answer}")
            print(f"Expected A : {expected_answer}")
            
            # Simple keyword matching as a proxy for answer relevance
            keywords = expected_answer.lower().split()
            match_score = sum(1 for kw in keywords if kw in answer.lower()) / len(keywords)
            
            print(f"Score (Keyword match proxy): {match_score:.2f}")
            
        except Exception as e:
            print(f"Evaluation failed for this question: {str(e)}")

if __name__ == "__main__":
    run_eval()
