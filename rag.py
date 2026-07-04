"""
Query The Little Prince RAG system.

Usage:
    python rag.py                    # interactive mode
    python rag.py "your question"    # single-shot mode
"""

import sys

import chromadb
from sentence_transformers import SentenceTransformer

import citations
import llm

COLLECTION_NAME = "little_prince"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 8

_embed_model: SentenceTransformer | None = None
_collection = None


# Open the ChromaDB collection once and reuse it for every query
def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path="./chroma_db")
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


# Load the embedding model once and reuse it for every query
def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def order_passages(passages: list[dict]) -> list[dict]:
    """Sort retrieved passages back into original document order (grouped by
    source, then by position within that source). Similarity search returns
    passages in relevance-rank order, which scrambles narrative sequence —
    bad for questions where the order events happen in matters (e.g. "which
    planet did he visit first")."""
    return sorted(passages, key=lambda p: (p["source"], p["chunk_index"]))


# Embed the query and fetch the k most similar passages from ChromaDB
def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    model = get_embed_model()
    embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
    results = get_collection().query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas"],
    )
    passages = [
        {
            "text": doc,
            "source": meta.get("source", "unknown"),
            "kind": meta.get("kind", "reference"),
            "chunk_index": meta.get("chunk_index", 0),
        }
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
    return order_passages(passages)


# Build the full prompt: label each passage with its citation and quoting
# rules, then lay out the instructions the model must follow
def build_prompt(query: str, passages: list[dict]) -> str:
    def label(p: dict) -> str:
        cite = citations.in_text(p["source"])
        kind_label = "Book text" if p["kind"] == "book" else f"Reference ({p['source']})"
        return f"{kind_label} — cite as {cite}"

    # Labeled "Excerpt" not "Passage" — the model echoes whatever word labels
    # the source blocks, so this avoids that word leaking into the answer.
    numbered = "\n\n".join(
        f"[Excerpt {i+1} — {label(p)}]\n{p['text']}" for i, p in enumerate(passages)
    )
    return (
        "You are an expert on Antoine de Saint-Exupéry's novella *The Little Prince*. "
        "Answer the user's question with genuine analysis and explanation, grounded in the provided excerpts. "
        "The excerpts come from two kinds of sources: 'Book text' (the novella itself) and "
        "'Reference' material (background, themes, character notes, and plot analysis). Rules:\n"
        "1. Explain and interpret — don't just list quotes. Give a real answer.\n"
        "2. Cite sources using proper APA 7th-edition in-text citations. Each excerpt tells you exactly "
        "which citation to use, e.g. (Saint-Exupéry, 1943) or (SparkNotes Editors, n.d.-a) — copy it "
        "verbatim wherever you rely on that excerpt. Never invent a citation — refer to sources only by "
        "their citation.\n"
        "3. Only quote directly, word-for-word, from excerpts marked 'Book text'; copy those words exactly.\n"
        "4. Never quote 'Reference' excerpts verbatim — restate their ideas in your own words and use them "
        "only to inform your explanation of themes, characters, and background.\n"
        "5. Do not invent facts or quotes not present in the excerpts.\n"
        "6. If the excerpts lack enough information, say so honestly.\n\n"
        f"<excerpts>\n{numbered}\n</excerpts>\n\n"
        f"Question: {query}\n\n"
        "Reminder: cite only with the APA citations given above (e.g. (Saint-Exupéry, 1943)). Do not "
        "number or refer to the excerpts themselves (no 'Excerpt 1', 'Passage 2', etc.) in your answer."
    )


# Retrieve passages, prompt the model, stream the answer to the console,
# then print the reference list for whatever was actually cited
def ask(query: str) -> None:
    passages = retrieve(query)
    prompt = build_prompt(query, passages)

    print("\nAnswer:\n")
    full_response = ""
    stream = llm.chat(messages=[{"role": "user", "content": prompt}], stream=True)
    for chunk in stream:
        piece = chunk["message"]["content"]
        full_response += piece
        print(piece, end="", flush=True)
    print("\n")

    references = citations.references_cited_in(full_response)
    if references:
        print("References:")
        for ref in references:
            print(f"- {ref}")
        print()


# Simple REPL: read a question, print an answer, repeat until the user quits
def interactive_loop() -> None:
    print("Little Prince RAG — type your question, or 'quit' to exit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break
        ask(query)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        interactive_loop()
