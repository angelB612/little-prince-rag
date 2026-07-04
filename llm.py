"""
Chat-completion backend, isolated behind one function.

Wraps Gemini's hosted API (free tier) so the app can run on Streamlit
Community Cloud or anywhere else without a local Ollama server. Nothing in
rag.py or app.py needs to change if this backend is swapped again later.
"""

import os

from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"

_client: genai.Client | None = None


# Create the Gemini client once, using the API key from the environment
def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and set it as an environment "
                "variable (or in .streamlit/secrets.toml as GEMINI_API_KEY)."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _split_system(messages: list[dict]) -> tuple[str | None, list[types.Content]]:
    """Gemini takes system instructions and turn history separately, and
    calls the assistant role "model" rather than "assistant"."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part(text=m["content"])],
        )
        for m in messages
        if m["role"] != "system"
    ]
    return system, contents


def chat(messages: list[dict], stream: bool = True):
    """Yield chat-completion chunks. Each chunk is a dict with
    chunk["message"]["content"] holding the next piece of text, matching
    Ollama's streaming response shape."""
    system, contents = _split_system(messages)
    config = types.GenerateContentConfig(system_instruction=system)

    if not stream:
        # Single request, single response — return the whole answer at once
        response = _get_client().models.generate_content(
            model=MODEL, contents=contents, config=config
        )
        return {"message": {"content": response.text}}

    # Streaming request — yield each piece of text as it arrives
    response = _get_client().models.generate_content_stream(
        model=MODEL, contents=contents, config=config
    )
    return (
        {"message": {"content": chunk.text}} for chunk in response if chunk.text
    )
