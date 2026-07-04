"""
Streamlit UI for the Little Prince RAG system.

Usage:
    streamlit run app.py
"""

import os

# Must be set before torch/sentence-transformers import — loading a model in
# Streamlit's script-runner thread can deadlock PyTorch's OpenMP pool on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re

import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb

import citations
import ingest
import llm
from rag import build_prompt, order_passages, TOP_K, COLLECTION_NAME, EMBED_MODEL


# Bold any quoted text in a passage that also appears in the model's answer,
# so the sidebar shows at a glance what was actually quoted
def highlight_quotes(passage: str, response: str) -> str:
    quotes = re.findall(r'"([^"]{10,})"', response)
    for quote in quotes:
        if quote in passage:
            passage = passage.replace(quote, f"**{quote}**")
    return passage

# Cached so the model is only loaded once per session, not on every rerun
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource(show_spinner=False)
def load_spellchecker():
    from spellchecker import SpellChecker
    return SpellChecker()

# Fix typos in the user's question before embedding it, so retrieval isn't
# thrown off by a misspelled word
def correct_query(query: str) -> str:
    spell = load_spellchecker()
    return " ".join(spell.correction(w) or w for w in query.lower().split())

# Open the ChromaDB collection holding the indexed book/reference passages
def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection(COLLECTION_NAME)

st.set_page_config(page_title="The Little Prince RAG", page_icon="⚘", layout="wide")

# Custom CSS to theme the page like old cream-colored book pages
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Lato:wght@300;400&display=swap');

    /* Seamless cream background */
    .stApp,
    [data-testid="stHeader"],
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stAppViewBlockContainer"],
    section.main > div,
    .stChatInput,
    [data-testid="stChatInputContainer"] {
        background-color: #f5f0e6 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ede8db !important;
        border-right: 1px solid #d8d0c0;
    }
    [data-testid="stSidebar"] * { color: #2c2c2c !important; }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #5a7a6a !important;
        font-family: 'Playfair Display', Georgia, serif !important;
    }

    /* Title */
    h1 {
        color: #1a1a1a !important;
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
    }

    /* Subtitle */
    [data-testid="stCaption"], .stCaptionContainer p {
        color: #7a9a8a !important;
        font-family: 'Lato', sans-serif !important;
        font-weight: 300;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-size: 1rem !important;
    }

    /* Body text */
    .stMarkdown p, .stMarkdown li {
        color: #2c2c2c !important;
        font-family: 'Lato', sans-serif !important;
        line-height: 1.7;
    }

    /* Hide avatars */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
    }

    /* User message */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #eae4d8 !important;
        border: 1px solid #d8d0c0 !important;
        border-radius: 12px;
    }

    /* Assistant message */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: #fff !important;
        border: 1px solid #e4dfd6 !important;
        border-radius: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    /* Chat input — all one colour */
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div:focus-within {
        background: #fff !important;
        border: 1px solid #d8d0c0 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
        outline: none !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea:focus {
        background: #fff !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        caret-color: #5a7a6a !important;
        font-family: 'Lato', sans-serif !important;
        font-size: 1.05rem !important;
    }

    /* Submit button */
    [data-testid="stChatInput"] button,
    [data-testid="stChatInput"] button:hover,
    [data-testid="stChatInput"] button:focus,
    [data-testid="stChatInput"] button:active {
        background-color: #5a7a6a !important;
        border-color: #5a7a6a !important;
        color: #fff !important;
    }

    /* Divider */
    hr { border-color: #ddd8ce !important; }

    /* Quoted highlights */
    strong { color: #5a7a6a !important; }
</style>
""", unsafe_allow_html=True)

# Decorative background stars/sparkles (purely cosmetic, no interaction)
title_col, img_col = st.columns([5, 1])
with title_col:
    st.markdown("""
<div style="position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;">
  <span style="position:absolute;top:6%;left:8%;color:#c8a430;font-size:18px;opacity:0.55;">✦</span>
  <span style="position:absolute;top:12%;left:22%;color:#c8a430;font-size:11px;opacity:0.45;">★</span>
  <span style="position:absolute;top:5%;left:42%;color:#c8a430;font-size:20px;opacity:0.5;">✦</span>
  <span style="position:absolute;top:9%;left:60%;color:#c8a430;font-size:13px;opacity:0.45;">★</span>
  <span style="position:absolute;top:4%;left:75%;color:#c8a430;font-size:22px;opacity:0.55;">✦</span>
  <span style="position:absolute;top:14%;left:88%;color:#c8a430;font-size:12px;opacity:0.4;">★</span>
  <span style="position:absolute;top:35%;left:5%;color:#c8a430;font-size:14px;opacity:0.4;">✦</span>
  <span style="position:absolute;top:50%;left:92%;color:#c8a430;font-size:18px;opacity:0.5;">✦</span>
  <span style="position:absolute;top:65%;left:3%;color:#c8a430;font-size:20px;opacity:0.45;">★</span>
  <span style="position:absolute;top:72%;left:48%;color:#c8a430;font-size:12px;opacity:0.4;">✦</span>
  <span style="position:absolute;top:80%;left:70%;color:#c8a430;font-size:18px;opacity:0.5;">★</span>
  <span style="position:absolute;top:88%;left:85%;color:#c8a430;font-size:14px;opacity:0.45;">✦</span>
  <span style="position:absolute;top:92%;left:30%;color:#c8a430;font-size:16px;opacity:0.4;">★</span>
</div>
""", unsafe_allow_html=True)

title_col, img_col = st.columns([5, 1])
with title_col:
    st.title("The Little Prince")
    st.caption("What is essential is invisible to the eye... until now.")
with img_col:
    st.image("little_prince_fox.png", use_container_width=True)

# Chat history and the passages behind the last answer persist across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_passages" not in st.session_state:
    st.session_state.last_passages = []

# Sidebar shows the source passages used to answer the most recent question
with st.sidebar:
    st.image("little_prince.png", use_container_width=True)
    st.header("Retrieved Passages")
    if st.session_state.last_passages:
        last_response = st.session_state.messages[-1]["content"] if st.session_state.messages else ""
        for i, passage in enumerate(st.session_state.last_passages, 1):
            label = "Book" if passage["kind"] == "book" else passage["source"]
            st.markdown(
                f'<div style="display:inline-block;background:#5a7a6a;color:#fff;'
                f'font-size:0.7rem;font-family:Lato,sans-serif;letter-spacing:1px;'
                f'text-transform:uppercase;padding:2px 10px;border-radius:20px;'
                f'margin-bottom:6px;">Passage {i} · {label}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(highlight_quotes(passage["text"], last_response))
            st.divider()
    else:
        st.caption("Passages from the book will appear here after your first question.")

# Print the reference list under an answer, if any citations were used
def render_references(references: list[str]) -> None:
    if not references:
        return
    st.markdown("**References**")
    for ref in references:
        st.markdown(f"- {ref}")


# Redraw the full chat history on every rerun (Streamlit reruns the whole
# script on each interaction, so past messages aren't kept on screen otherwise)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_references(msg.get("references", []))

with st.spinner("Preparing your reading companion..."):
    load_embed_model()

# Reuse the prebuilt local index if there is one; otherwise (e.g. deployed
# with no data/chroma_db committed) let the user upload their own copy.
if "collection" not in st.session_state:
    try:
        st.session_state.collection = get_collection()
    except Exception:
        st.session_state.collection = None

if st.session_state.collection is None:
    st.info("No book index found. Upload your own copy of the book to get started.")
    book_file = st.file_uploader("The Little Prince (PDF or TXT)", type=["pdf", "txt"])
    reference_files = st.file_uploader(
        "Optional: reference/study material (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    if book_file:
        with st.spinner("Building the index from your upload..."):
            st.session_state.collection = ingest.build_session_collection(
                book_file, reference_files, load_embed_model()
            )
        st.rerun()
    st.stop()

query = st.chat_input("What would you like to know?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    model = load_embed_model()
    corrected = correct_query(query)

    # Embed text and fetch the closest passages, with their similarity distances
    def query_collection(text: str):
        emb = model.encode([text], normalize_embeddings=True)[0].tolist()
        return st.session_state.collection.query(
            query_embeddings=[emb],
            n_results=TOP_K,
            include=["documents", "distances", "metadatas"],
        )

    # Try the query alone first so a topic-shift isn't dragged toward the
    # last topic; only retry with recent context if it looks like a weak follow-up.
    results = query_collection(corrected)
    best_distance = min(results["distances"][0])

    if best_distance > 0.7:
        recent_context = " ".join(m["content"] for m in st.session_state.messages[-5:-1])
        if recent_context:
            context_results = query_collection(f"{recent_context} {corrected}".strip())
            context_distance = min(context_results["distances"][0])
            if context_distance < best_distance:
                results, best_distance = context_results, context_distance

    passages = order_passages([
        {
            "text": doc,
            "source": meta.get("source", "unknown"),
            "kind": meta.get("kind", "reference"),
            "chunk_index": meta.get("chunk_index", 0),
        }
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ])

    # Nothing retrieved was actually close to the question — treat it as off-topic
    if best_distance > 0.7:
        with st.chat_message("assistant"):
            st.markdown("I can only answer questions about *The Little Prince*.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "I can only answer questions about *The Little Prince*."
        })
        st.rerun()

    prompt = build_prompt(query, passages)

    # Stream the answer into the placeholder, then append it to chat history
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("*Thinking...*")
        full_response = ""

        system = (
            "You are a reading assistant exclusively for the book 'The Little Prince'. "
            "You must ONLY answer questions about this book using the provided excerpts. "
            "If a question is not about the book, or cannot be answered from the excerpts, "
            "respond only with: 'I can only answer questions about The Little Prince.' "
            "Never use outside knowledge. Never answer math, general knowledge, or unrelated questions. "
            "Excerpts marked 'Book text' may be quoted verbatim; excerpts marked 'Reference' "
            "(background, themes, character notes, plot analysis) must never be quoted "
            "word-for-word — paraphrase them in your own words to inform your answer. "
            "Cite every excerpt you rely on with the exact APA 7th-edition in-text citation given "
            "alongside it (e.g. (Saint-Exupéry, 1943)). Do not number or refer to the excerpts "
            "themselves in your answer."
        )
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        stream = llm.chat(
            messages=[{"role": "system", "content": system}] + history + [{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            full_response += chunk["message"]["content"]
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)
        references = citations.references_cited_in(full_response)
        render_references(references)

    st.session_state.last_passages = passages
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "references": references,
    })
    st.rerun()
