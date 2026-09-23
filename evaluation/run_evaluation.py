"""
VoicePDF Evaluation Runner — Evaluates RAG pipeline quality using benchmark datasets.

Phase 7: Evaluation & Quality Improvements.

Metrics:
  - Answer Relevance: Keyword overlap between generated and expected answers.
  - Context Precision: Whether retrieved context contains expected keywords.
  - Faithfulness Proxy: Whether the answer can be traced to the provided context.

Usage:
  python -m evaluation.run_evaluation
"""
import json
import os
import sys
import logging

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")


def load_dataset(path: str = None) -> list[dict]:
    """Load the evaluation dataset from JSON."""
    path = path or DATASET_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_keyword_overlap(text: str, keywords: str) -> float:
    """
    Calculate a simple keyword overlap score between text and expected keywords.
    Returns a float between 0.0 and 1.0.
    """
    if not text or not keywords:
        return 0.0

    text_lower = text.lower()
    keyword_tokens = [kw.strip().lower() for kw in keywords.split() if kw.strip()]

    if not keyword_tokens:
        return 0.0

    matches = sum(1 for kw in keyword_tokens if kw in text_lower)
    return matches / len(keyword_tokens)


def evaluate_single(question: str, generated_answer: str, expected_answer: str,
                    retrieved_context: str, expected_context: str) -> dict:
    """
    Evaluate a single Q&A pair with multiple metrics.

    Returns:
        dict with metric scores.
    """
    # Answer Relevance: Does the generated answer contain expected keywords?
    answer_relevance = calculate_keyword_overlap(generated_answer, expected_answer)

    # Context Precision: Does the retrieved context contain the expected context keywords?
    context_precision = calculate_keyword_overlap(retrieved_context, expected_context)

    # Faithfulness Proxy: Are the key terms in the answer also found in the context?
    # (Simple heuristic: answer words that appear in context vs total answer words)
    answer_words = set(generated_answer.lower().split())
    context_words = set(retrieved_context.lower().split()) if retrieved_context else set()
    if answer_words:
        faithfulness = len(answer_words & context_words) / len(answer_words)
    else:
        faithfulness = 0.0

    return {
        "answer_relevance": round(answer_relevance, 3),
        "context_precision": round(context_precision, 3),
        "faithfulness": round(faithfulness, 3),
    }


def run_eval(document_ids: list[str] = None, dataset_path: str = None) -> dict:
    """
    Run the full evaluation pipeline against the RAG system.

    Args:
        document_ids: Optional list of document IDs to scope retrieval.
        dataset_path: Optional path to evaluation dataset JSON.

    Returns:
        dict with per-question results and aggregate scores.
    """
    from rag.rag_service import generate_answer
    from rag.context import build_context_string
    from rag.retriever import get_retriever

    dataset = load_dataset(dataset_path)
    results = []

    print(f"\n{'='*70}")
    print(f" VoicePDF RAG Evaluation — {len(dataset)} questions")
    print(f"{'='*70}\n")

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        expected_answer = item.get("expected_answer", "")
        expected_context = item.get("expected_context", "")

        print(f"Q{i}: {question}")

        try:
            # Get answer from RAG pipeline
            answer, citations = generate_answer(question, document_ids=document_ids)

            # Get retrieved context for evaluation
            retriever = get_retriever(document_ids=document_ids)
            docs = retriever.invoke(question)
            context_str = build_context_string(docs) if docs else ""

            # Evaluate
            scores = evaluate_single(
                question=question,
                generated_answer=answer,
                expected_answer=expected_answer,
                retrieved_context=context_str,
                expected_context=expected_context,
            )

            result = {
                "question": question,
                "generated_answer": answer[:200],  # Truncate for report
                "citations": citations,
                "scores": scores,
                "status": "success",
            }

            print(f"   Answer: {answer[:100]}...")
            print(f"   Scores: Relevance={scores['answer_relevance']:.2f}  "
                  f"Context={scores['context_precision']:.2f}  "
                  f"Faithfulness={scores['faithfulness']:.2f}")

        except Exception as e:
            result = {
                "question": question,
                "generated_answer": "",
                "citations": [],
                "scores": {"answer_relevance": 0, "context_precision": 0, "faithfulness": 0},
                "status": f"error: {str(e)}",
            }
            print(f"   ERROR: {e}")

        results.append(result)
        print()

    # Aggregate scores
    successful = [r for r in results if r["status"] == "success"]
    if successful:
        avg_relevance = sum(r["scores"]["answer_relevance"] for r in successful) / len(successful)
        avg_context = sum(r["scores"]["context_precision"] for r in successful) / len(successful)
        avg_faith = sum(r["scores"]["faithfulness"] for r in successful) / len(successful)
    else:
        avg_relevance = avg_context = avg_faith = 0.0

    summary = {
        "total_questions": len(dataset),
        "successful": len(successful),
        "failed": len(dataset) - len(successful),
        "avg_answer_relevance": round(avg_relevance, 3),
        "avg_context_precision": round(avg_context, 3),
        "avg_faithfulness": round(avg_faith, 3),
    }

    print(f"{'='*70}")
    print(f" EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total Questions:      {summary['total_questions']}")
    print(f"  Successful:           {summary['successful']}")
    print(f"  Failed:               {summary['failed']}")
    print(f"  Avg Answer Relevance: {summary['avg_answer_relevance']:.3f}")
    print(f"  Avg Context Precision:{summary['avg_context_precision']:.3f}")
    print(f"  Avg Faithfulness:     {summary['avg_faithfulness']:.3f}")
    print(f"{'='*70}\n")

    return {"summary": summary, "results": results}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run VoicePDF RAG Evaluation")
    parser.add_argument("--dataset", type=str, default=None, help="Path to evaluation dataset JSON")
    parser.add_argument("--doc-ids", nargs="*", default=None, help="Document IDs to scope retrieval")
    args = parser.parse_args()

    run_eval(document_ids=args.doc_ids, dataset_path=args.dataset)
