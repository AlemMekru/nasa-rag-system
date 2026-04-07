#!/usr/bin/env python3
"""
NASA RAG Chat with RAGAS Evaluation Integration

Enhanced version of the simple RAG chat that includes real-time evaluation
and feedback collection for continuous improvement.
"""

import streamlit as st
import os
import json

import ragas_evaluator
import rag_client
import llm_client

from typing import Dict, List, Optional
from embedding_pipeline import ChromaEmbeddingPipelineTextOnly
# RAGAS imports
try:
    import ragas_evaluator
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="NASA RAG Chat with Evaluation",
    page_icon="🚀",
    layout="wide"
)


@st.cache_data
def load_evaluation_dataset() -> List[Dict]:
    """Load the evaluation dataset from file"""
    try:
        with open("evaluation_dataset.txt", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def get_relevant_doc_ids(question: str, dataset: List[Dict]) -> List[str]:
    """Look up relevant_doc_ids for a question in the evaluation dataset"""
    for item in dataset:
        if item.get("question", "").strip().lower() == question.strip().lower():
            return item.get("relevant_doc_ids", [])
    return []


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    return rag_client.discover_chroma_backends()


@st.cache_resource
def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""
    try:
        collection = rag_client.initialize_rag_system(chroma_dir, collection_name)
        return collection, True, None
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(collection, openai_key: str, query: str, n_results: int = 3,
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    try:
        embedding_function = lambda q: llm_client.get_embedding(openai_key, q)

        return rag_client.retrieve_documents(
            collection,
            query,
            embedding_function,
            n_results,
            mission_filter
        )
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return None


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    return rag_client.format_context(documents, metadatas)


def generate_response(openai_key, user_message: str, context: str,
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""
    try:
        return llm_client.generate_response(
            openai_key,
            user_message,
            context,
            conversation_history,
            model
        )
    except Exception as e:
        return f"Error generating response: {e}"


def evaluate_response_quality(
    openai_key: str,
    question: str,
    answer: str,
    contexts: List[str],
    retrieved_doc_ids: Optional[List[str]] = None,
    relevant_doc_ids: Optional[List[str]] = None,
    selected_metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Evaluate response quality using configurable metrics"""
    try:
        return ragas_evaluator.evaluate_response_quality(
            question=question,
            answer=answer,
            contexts=contexts,
            openai_key=openai_key,
            selected_metrics=selected_metrics,
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=relevant_doc_ids,
        )
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}


def display_evaluation_metrics(scores: Dict[str, float]):
    """Display evaluation metrics in the sidebar"""
    if "error" in scores:
        st.sidebar.error(f"Evaluation Error: {scores['error']}")
        return

    st.sidebar.subheader("📊 Response Quality")

    for metric_name, score in scores.items():
        if isinstance(score, (int, float)):
            st.sidebar.metric(
                label=metric_name.replace('_', ' ').title(),
                value=f"{score:.3f}"
            )
            st.sidebar.progress(max(0.0, min(float(score), 1.0)))


def main():
    st.title("🚀 NASA Space Mission Chat with Evaluation")
    st.markdown("Chat with AI about NASA space missions with real-time quality evaluation")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_backend" not in st.session_state:
        st.session_state.current_backend = None
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []

    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")

        with st.spinner("Discovering ChromaDB backends..."):
            available_backends = discover_chroma_backends()

        if not available_backends:
            st.error("No ChromaDB backends found!")
            st.info("Please run the embedding pipeline first.")
            st.stop()

        st.subheader("📊 ChromaDB Backend")
        backend_options = {k: v["display_name"] for k, v in available_backends.items()}

        selected_backend_key = st.selectbox(
            "Select Document Collection",
            options=list(backend_options.keys()),
            format_func=lambda x: backend_options[x],
            help="Choose which document collection to use for retrieval"
        )

        selected_backend = available_backends[selected_backend_key]

        st.subheader("🔑 OpenAI Settings")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key"
        )

        if not openai_key:
            st.warning("Please enter your OpenAI API key")
            st.stop()

        st.subheader("🤖 Model Settings")
        model_choice = st.selectbox(
            "OpenAI Model",
            options=["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"],
            index=0,
            help="Choose the OpenAI model for responses"
        )

        st.subheader("🔍 Retrieval Settings")
        n_docs = st.slider("Documents to retrieve", 1, 10, 3)

        mission_filter = st.selectbox(
            "Mission Filter",
            options=["all", "apollo_11", "apollo_13", "challenger", "unknown"],
            index=0,
            help="Optionally restrict retrieval to one mission"
        )

        st.subheader("📊 Evaluation Settings")
        enable_evaluation = st.checkbox("Enable RAGAS Evaluation", value=RAGAS_AVAILABLE)

        if st.session_state.current_backend != selected_backend_key:
            st.session_state.current_backend = selected_backend_key
            st.cache_resource.clear()

    # Initialize RAG system
    with st.spinner("Initializing RAG system..."):
        collection, success, error = initialize_rag_system(
            selected_backend["directory"],
            selected_backend["collection_name"]
        )

    if not success:
        st.error(f"Failed to initialize RAG system: {error}")
        st.stop()

    # Display evaluation metrics if available
    if st.session_state.last_evaluation and enable_evaluation:
        display_evaluation_metrics(st.session_state.last_evaluation)

    # Display last retrieved contexts
    if st.session_state.last_contexts:
        with st.expander("📚 Last Retrieved Context Chunks"):
            for i, ctx in enumerate(st.session_state.last_contexts, start=1):
                st.markdown(f"**Chunk {i}**")
                st.write(ctx[:1200] + ("..." if len(ctx) > 1200 else ""))

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input

    if prompt := st.chat_input("Ask about NASA space missions..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                docs_result = retrieve_documents(
                    collection,
                    openai_key,
                    prompt,
                    n_docs,
                    mission_filter
                )

                st.write("DEBUG docs_result:", docs_result)

                context = ""

                contexts_list = []

                if isinstance(docs_result, dict) and "documents" in docs_result:
                    documents = docs_result.get("documents", [])
                    if documents and len(documents) > 0 and documents[0]:
                        contexts_list = documents[0]

                    context = "\n\n".join(contexts_list) if contexts_list else ""
                    #st.write("Retrieved context count:", len(contexts_list))

                    st.session_state.last_contexts = contexts_list

                response = generate_response(
                    openai_key,
                    prompt,
                    context,
                    st.session_state.messages[:-1],
                    model_choice
                )
                st.markdown(response)

                if enable_evaluation and RAGAS_AVAILABLE and contexts_list:
                    with st.spinner("Evaluating response quality..."):
                        retrieved_ids = []
                        if docs_result and docs_result.get("ids") and len(docs_result["ids"]) > 0:
                            retrieved_ids = docs_result["ids"][0]

                        # Load evaluation dataset and get relevant_doc_ids for this question
                        eval_dataset = load_evaluation_dataset()
                        relevant_ids = get_relevant_doc_ids(prompt, eval_dataset)

                        evaluation_scores = evaluate_response_quality(
                            openai_key=openai_key,
                            question=prompt,
                            answer=response,
                            contexts=contexts_list,
                            selected_metrics=["faithfulness", "response_relevancy", "precision"],
                            retrieved_doc_ids=retrieved_ids,
                            relevant_doc_ids=relevant_ids
                        )
                        st.session_state.last_evaluation = evaluation_scores
                elif enable_evaluation and not contexts_list:
                    st.session_state.last_evaluation = {"error": "No retrieved context available for evaluation."}

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()