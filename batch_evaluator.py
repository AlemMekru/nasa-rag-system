#!/usr/bin/env python3
"""
Batch Evaluation Runner for NASA RAG System

Runs end-to-end evaluation on the entire evaluation_dataset.txt test set.
For each question, performs retrieval, answer generation, and evaluation,
then outputs per-question results and aggregate metric summaries.
"""

import json
import os
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import project modules
import rag_client
import llm_client
import ragas_evaluator
from embedding_pipeline import ChromaEmbeddingPipelineTextOnly


def load_evaluation_dataset(dataset_path: str) -> List[Dict]:
    """Load the evaluation dataset from JSON file"""
    try:
        with open(dataset_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading evaluation dataset: {e}")
        return []


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend"""
    try:
        collection = rag_client.initialize_rag_system(chroma_dir, collection_name)
        return collection, True, None
    except Exception as e:
        return None, False, str(e)


def run_single_evaluation(
    question: str,
    expected_answer: str,
    collection,
    openai_key: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None,
    model: str = "gpt-3.5-turbo"
) -> Dict[str, Any]:
    """
    Run the complete evaluation pipeline for a single question:
    1. Retrieve documents
    2. Generate answer
    3. Evaluate quality
    """
    try:
        # Step 1: Retrieve documents
        pipeline = ChromaEmbeddingPipelineTextOnly(openai_api_key=openai_key)
        docs_result = rag_client.retrieve_documents(
            collection,
            openai_key,
            question,
            n_results,
            mission_filter
        )

        contexts_list = []
        retrieved_ids = []

        if isinstance(docs_result, dict) and "documents" in docs_result:
            documents = docs_result.get("documents", [])
            if documents and len(documents) > 0 and documents[0]:
                contexts_list = documents[0]

            if docs_result.get("ids") and len(docs_result["ids"]) > 0:
                retrieved_ids = docs_result["ids"][0]

        # Step 2: Generate answer
        context = "\n\n".join(contexts_list) if contexts_list else ""
        generated_answer = llm_client.generate_response(
            openai_key,
            question,
            context,
            [],  # No conversation history for batch evaluation
            model
        )

        # Step 3: Evaluate quality
        evaluation_scores = ragas_evaluator.evaluate_response_quality(
            question=question,
            answer=generated_answer,
            contexts=contexts_list,
            openai_key=openai_key,
            selected_metrics=["faithfulness", "response_relevancy", "precision"],
            retrieved_doc_ids=retrieved_ids,
            relevant_doc_ids=[]  # Will be set from dataset
        )

        return {
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "contexts_count": len(contexts_list),
            "retrieved_doc_ids": retrieved_ids,
            "evaluation_scores": evaluation_scores,
            "success": True
        }

    except Exception as e:
        return {
            "question": question,
            "expected_answer": expected_answer,
            "error": str(e),
            "success": False
        }


def run_batch_evaluation(
    dataset: List[Dict],
    collection,
    openai_key: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run batch evaluation on the entire dataset and compute aggregate metrics
    """
    results = []
    aggregate_metrics = {
        "faithfulness": [],
        "response_relevancy": [],
        "precision": []
    }

    print(f"Starting batch evaluation of {len(dataset)} questions...")

    for i, item in enumerate(dataset, 1):
        question = item.get("question", "")
        expected_answer = item.get("answer", "")
        relevant_doc_ids = item.get("relevant_doc_ids", [])

        print(f"[{i}/{len(dataset)}] Evaluating: {question[:60]}{'...' if len(question) > 60 else ''}")

        # Run evaluation
        result = run_single_evaluation(
            question=question,
            expected_answer=expected_answer,
            collection=collection,
            openai_key=openai_key,
            n_results=n_results,
            mission_filter=mission_filter,
            model=model
        )

        # Override relevant_doc_ids with dataset ground truth
        if result["success"] and "evaluation_scores" in result:
            result["evaluation_scores"]["relevant_doc_ids"] = relevant_doc_ids

            # Recalculate precision with correct ground truth
            if "retrieved_doc_ids" in result:
                precision = ragas_evaluator.compute_precision(
                    result["retrieved_doc_ids"],
                    relevant_doc_ids
                )
                result["evaluation_scores"]["precision"] = precision

        results.append(result)

        # Collect metrics for aggregation
        if result["success"] and "evaluation_scores" in result:
            scores = result["evaluation_scores"]
            for metric in aggregate_metrics.keys():
                if metric in scores and isinstance(scores[metric], (int, float)):
                    aggregate_metrics[metric].append(scores[metric])

    # Calculate aggregate statistics
    summary = {}
    for metric, values in aggregate_metrics.items():
        if values:
            summary[metric] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "values": values
            }
        else:
            summary[metric] = {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "values": []}

    # Save results if output file specified
    if output_file:
        output_data = {
            "summary": summary,
            "results": results,
            "metadata": {
                "total_questions": len(dataset),
                "successful_evaluations": len([r for r in results if r["success"]]),
                "chroma_collection": str(collection) if collection else None,
                "model": model,
                "n_results": n_results,
                "mission_filter": mission_filter
            }
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return {
        "summary": summary,
        "results": results
    }


def print_evaluation_report(results: Dict[str, Any]):
    """Print a formatted evaluation report"""
    summary = results["summary"]
    individual_results = results["results"]

    print("\n" + "="*80)
    print("BATCH EVALUATION REPORT")
    print("="*80)

    print(f"\nTotal Questions: {len(individual_results)}")
    successful = len([r for r in individual_results if r["success"]])
    print(f"Successful Evaluations: {successful}")
    print(".1f")

    print("\n" + "-"*80)
    print("AGGREGATE METRICS SUMMARY")
    print("-"*80)

    for metric, stats in summary.items():
        if stats["count"] > 0:
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(".3f")
            print(".3f")
            print(".3f")
            print(f"  Sample Size: {stats['count']}")

    print("\n" + "-"*80)
    print("PER-QUESTION RESULTS")
    print("-"*80)

    for i, result in enumerate(individual_results, 1):
        print(f"\n{i}. {result['question'][:80]}{'...' if len(result['question']) > 80 else ''}")

        if result["success"]:
            scores = result["evaluation_scores"]
            contexts_count = result.get("contexts_count", 0)

            print(f"   Contexts Retrieved: {contexts_count}")
            print(f"   Faithfulness: {scores.get('faithfulness', 0.0):.3f}")
            print(f"   Response Relevancy: {scores.get('response_relevancy', 0.0):.3f}")
            print(f"   Precision: {scores.get('precision', 0.0):.3f}")
        else:
            print(f"   ERROR: {result.get('error', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(description="Run batch evaluation on NASA RAG system")
    parser.add_argument(
        "--chroma-dir",
        required=True,
        help="Path to ChromaDB directory"
    )
    parser.add_argument(
        "--collection-name",
        required=True,
        help="Name of ChromaDB collection"
    )
    parser.add_argument(
        "--openai-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--dataset-path",
        default="evaluation_dataset.txt",
        help="Path to evaluation dataset JSON file"
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=3,
        help="Number of documents to retrieve per question"
    )
    parser.add_argument(
        "--mission-filter",
        choices=["all", "apollo_11", "apollo_13", "challenger", "unknown"],
        default="all",
        help="Filter retrieval to specific mission"
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        choices=["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"],
        help="OpenAI model to use for answer generation"
    )
    parser.add_argument(
        "--output",
        help="Output file for detailed results (JSON format)"
    )

    args = parser.parse_args()

    if not args.openai_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable or use --openai-key")
        return 1

    # Load evaluation dataset
    print(f"Loading evaluation dataset from: {args.dataset_path}")
    dataset = load_evaluation_dataset(args.dataset_path)
    if not dataset:
        print("Error: Could not load evaluation dataset")
        return 1

    print(f"Loaded {len(dataset)} evaluation questions")

    # Initialize RAG system
    print(f"Initializing RAG system with ChromaDB: {args.chroma_dir}")
    print(f"Collection: {args.collection_name}")

    collection, success, error = initialize_rag_system(args.chroma_dir, args.collection_name)
    if not success:
        print(f"Error initializing RAG system: {error}")
        return 1

    # Run batch evaluation
    results = run_batch_evaluation(
        dataset=dataset,
        collection=collection,
        openai_key=args.openai_key,
        n_results=args.n_results,
        mission_filter=None if args.mission_filter == "all" else args.mission_filter,
        model=args.model,
        output_file=args.output
    )

    # Print report
    print_evaluation_report(results)

    return 0



if __name__ == "__main__":
    exit(main())
