import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for ChromaDB directories
    candidate_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and ("chroma" in d.name.lower() or "db" in d.name.lower())
    ]

    # Loop through each discovered directory
    for directory in candidate_dirs:
        # Wrap connection attempt in try-except block for error handling
        try:
            # Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(path=str(directory))

            # Retrieve list of available collections from the database
            collections = client.list_collections()

            # Loop through each collection found
            for collection_obj in collections:
                collection_name = (
                    collection_obj.name if hasattr(collection_obj, "name")
                    else str(collection_obj)
                )

                # Create unique identifier key combining directory and collection names
                backend_key = f"{directory.name}:{collection_name}"

                try:
                    collection = client.get_collection(collection_name)
                    document_count = collection.count()
                except Exception:
                    document_count = "unknown"

                # Build information dictionary
                backend_info = {
                    "directory": str(directory),
                    "collection_name": collection_name,
                    "display_name": f"{directory.name} / {collection_name} ({document_count} docs)",
                    "document_count": document_count,
                }

                # Add collection information to backends dictionary
                backends[backend_key] = backend_info

        # Handle connection or access errors gracefully
        except Exception as e:
            # Create fallback entry for inaccessible directories
            error_message = str(e)
            if len(error_message) > 60:
                error_message = error_message[:60] + "..."

            backend_key = f"{directory.name}:unavailable"

            # Include error information in display name with truncation
            backends[backend_key] = {
                "directory": str(directory),
                "collection_name": "unavailable",
                "display_name": f"{directory.name} / unavailable ({error_message})",
                "document_count": "unknown",
            }

    # Return complete backends dictionary with all discovered collections
    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    # Create a chomadb persistentclient
    client = chromadb.PersistentClient(path=chroma_dir)

    # Return the collection with the collection_name
    return client.get_collection(collection_name)


from typing import Optional, Dict, Callable, List

def retrieve_documents(
    collection,
    query: str,
    embedding_function: Callable[[str], List[float]],
    n_results: int = 3,
    mission_filter: Optional[str] = None
) -> Optional[Dict]:
    try:
        print("DEBUG query:", query)
        print("DEBUG embedding_function type:", type(embedding_function))
        print("DEBUG embedding_function name:", getattr(embedding_function, "__name__", "no_name"))

        where_filter = None
        if mission_filter and mission_filter.strip().lower() != "all":
            where_filter = {"mission": mission_filter.strip().lower()}

        query_embedding = embedding_function(query)

        print("DEBUG query_embedding type:", type(query_embedding))
        print("DEBUG query_embedding length:", len(query_embedding) if query_embedding else 0)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        print("DEBUG results keys:", results.keys() if isinstance(results, dict) else type(results))
        print("DEBUG documents present:", "documents" in results if isinstance(results, dict) else False)

        return results

    except Exception as e:
        print("DEBUG retrieve_documents error:", str(e))
        return {"error": str(e)}


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    # Initialize list with header text for context section
    context_parts = ["Retrieved Context:\n"]

    # Loop through paired documents and their metadata using enumeration
    for idx, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
        # Extract mission information from metadata with fallback value
        mission = metadata.get("mission", "unknown")

        # Clean up mission name formatting (replace underscores, capitalize)
        mission = mission.replace("_", " ").title()

        # Extract source information from metadata with fallback value
        source = metadata.get("source", "unknown source")

        # Extract category information from metadata with fallback value
        category = metadata.get("document_category", "general")

        # Clean up category name formatting (replace underscores, capitalize)
        category = category.replace("_", " ").title()

        # Create formatted source header with index number and extracted information
        source_header = f"[Source {idx}] Mission: {mission} | Source: {source} | Category: {category}"

        # Add source header to context parts list
        context_parts.append(source_header)

        # Check document length and truncate if necessary
        if len(document) > 800:
            document_text = document[:800] + "..."
        else:
            document_text = document

        # Add truncated or full document content to context parts list
        context_parts.append(document_text)
        context_parts.append("")

    # Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)