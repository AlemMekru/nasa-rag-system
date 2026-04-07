from typing import Dict, List, Optional, Any

from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import ResponseRelevancy, Faithfulness
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


AVAILABLE_METRICS = {
    "faithfulness",
    "response_relevancy",
    "precision",
}


def compute_precision(
    retrieved_doc_ids: Optional[List[str]],
    relevant_doc_ids: Optional[List[str]],
) -> float:
    """
    Precision = relevant retrieved / total retrieved
    """
    retrieved_set = set(retrieved_doc_ids or [])
    relevant_set = set(relevant_doc_ids or [])

    if not retrieved_set:
        return 0.0

    return len(retrieved_set & relevant_set) / len(retrieved_set)


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    openai_key: Optional[str] = None,
    selected_metrics: Optional[List[str]] = None,
    retrieved_doc_ids: Optional[List[str]] = None,
    relevant_doc_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate response quality using configurable metrics.

    Supported metrics:
    - faithfulness
    - response_relevancy
    - precision
    """
    if selected_metrics is None:
        selected_metrics = ["faithfulness", "response_relevancy", "precision"]

    invalid_metrics = [m for m in selected_metrics if m not in AVAILABLE_METRICS]
    if invalid_metrics:
        return {
            "error": f"Unsupported metrics: {invalid_metrics}",
            "available_metrics": sorted(AVAILABLE_METRICS),
        }

    results: Dict[str, Any] = {}

    try:
        needs_ragas = any(
            metric in selected_metrics
            for metric in ["faithfulness", "response_relevancy"]
        )

        if needs_ragas:
            evaluator_llm = LangchainLLMWrapper(
                ChatOpenAI(
                    model="gpt-3.5-turbo",
                    api_key=openai_key,
                    temperature=0,
                )
            )

            evaluator_embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    api_key=openai_key,
                )
            )

            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )

            if "faithfulness" in selected_metrics:
                faithfulness_metric = Faithfulness(llm=evaluator_llm)
                faithfulness_score = faithfulness_metric.single_turn_score(sample)
                results["faithfulness"] = float(faithfulness_score)

            if "response_relevancy" in selected_metrics:
                response_relevancy_metric = ResponseRelevancy(
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                )
                response_relevancy_score = response_relevancy_metric.single_turn_score(sample)
                results["response_relevancy"] = float(response_relevancy_score)

        if "precision" in selected_metrics:
            results["precision"] = compute_precision(
                retrieved_doc_ids=retrieved_doc_ids,
                relevant_doc_ids=relevant_doc_ids,
            )

        results["selected_metrics"] = selected_metrics
        return results

    except Exception as e:
        return {"error": str(e)}