from typing import Dict, List
from openai import OpenAI

def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo"
) -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt (VERY IMPORTANT for rubric)
    system_prompt = """
You are a NASA mission expert assistant.

Your job is to answer questions ONLY using the provided context from NASA mission documents.

Rules:
- Use ONLY the provided context.
- Do NOT make up facts.
- If the answer is not in the context, say: "I don't have enough information from the provided documents."
- Be clear, factual, and concise.
- When possible, reference the mission and source.
"""

    # Create OpenAI Client
    client = OpenAI(api_key=openai_key)

    # Set context in messages
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Add conversation history
    for turn in conversation_history:
        if "role" in turn and "content" in turn:
            messages.append({
                "role": turn["role"],
                "content": turn["content"]
            })

    # Add current context + user question
    user_prompt = f"""
Context:
{context}

Question:
{user_message}
"""

    messages.append({
        "role": "user",
        "content": user_prompt
    })

    # Send request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2
    )
    # Return response
    return response.choices[0].message.content.strip()

def get_embedding(openai_key: str, text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Get embedding for query (used in RAG retrieval)"""
    client = OpenAI(api_key=openai_key)

    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response.data[0].embedding