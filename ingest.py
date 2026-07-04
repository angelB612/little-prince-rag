"""
Ingest The Little Prince and its supporting study materials into a ChromaDB vector store.

Usage:
    python ingest.py                       # ingest every .pdf/.txt file in ./data
    python ingest.py --data-dir mydata     # ingest a different directory

Each file is tagged with its source filename and a "kind":
    "book"      — the novella itself (TheLittlePrince.pdf); safe to quote verbatim
    "reference" — background/themes/character-list/analysis material; used only
                  to ground and inform answers, never to be quoted word-for-word

Accepts PDF or plain UTF-8 text files.
"""

import argparse
import re
import sys
from pathlib import Path

import chromadb
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "little_prince"
CHUNK_SIZE = 400        # target characters per chunk
CHUNK_OVERLAP = 80      # overlap to preserve context across chunk boundaries
MODEL_NAME = "all-MiniLM-L6-v2"
DATA_DIR_DEFAULT = "data"
BOOK_FILENAME = "TheLittlePrince.pdf"


# Recurring nav/chrome blocks from study-guide PDFs (SparkNotes, Britannica),
# stripped before line cleanup so nav fragments aren't read as book content.
_MASTHEAD_RE = re.compile(
    r"PLUS\nThe Little Prince\nAntoine de Saint-Exupéry\nStudy Guide\n.+\n"
    r"Start free trial\nLog in\nNext\n"
)
_FOOTER_NAV_RE = re.compile(
    r"\n?Summary\nCharacters\nLiterary Devices\nQuestions & Answers\n"
    r"Quotes\nQuick Quizzes\nDeeper Study\n?"
)
_BRITANNICA_TOPNAV_RE = re.compile(
    r"Games & Quizzes\nHistory & Society\nScience & Tech\nBiographies\n"
    r"Animals & Nature\nGeography & Travel\nArts & Culture\n.*?CITE more_vert\n",
    re.S,
)
_POPULAR_PAGES_RE = re.compile(
    r"Popular pages:.*?QUIZ: Which Greek God Are You\?\n?", re.S
)
# Everything after "Next section" on a SparkNotes-style page is trailing
# nav/upsell chrome, never article content.
_TRAILING_CHROME_RE = re.compile(r"\nNext section\b.*", re.S)
_READ_MORE_ABOUT_RE = re.compile(r"Read more about[^\n]*\n[^\n]*\n?", re.I)
_READ_MORE_COLON_RE = re.compile(r"^Read more:.*$\n?", re.I | re.M)
_READ_INDEPTH_RE = re.compile(r"^Read an in-depth analysis of.*$\n?", re.I | re.M)
_EMOJI_RE = re.compile("[\U0001F000-\U0001FFFF☀-➿←-⇿⬀-⯿]")

# Google Material Icons appear as bare ligature words (e.g. "zoom_in") in
# PDF-exported pages — never real prose, so a line matching one is dropped.
_ICON_LIGATURES = {
    "keyboard_arrow_right", "keyboard_arrow_down", "expand_more", "more_vert",
    "account_circle", "verified_user", "zoom_in", "menusearch", "today",
}

# One-off nav/ad/byline lines from the SparkNotes/Britannica exports in
# data/ — extend if new reference sources add different site chrome.
_BOILERPLATE_LINES = {
    "plus", "study guide", "start free trial", "log in", "sign up",
    "notes", "see all notes", "add note with sparknotes plus",
    "add your thoughts right here!", "first name", "last name", "email",
    "sign up for our latest news and updates!",
    "by entering your email address you agree to receive emails",
    "from sparknotes and verify that you are over the age of 13.",
    "you can view our privacy policy here. unsubscribe from our",
    "emails at any time.",
    "print", "cite more_vert", "britannica ai", "ask anything",
    "quick summary", "toc table of contents", "top questions",
    "expand_moreshow more", "britannica quiz", "classic children’s books quiz",
    "see all related content", "subscribe", "subscribe now",
    "save 30% off a britannica subscription! subscribe now",
    "this article was most recently revised and updated by encyclopaedia britannica.",
    "literature keyboard_arrow_right novels & short stories",
    "literature keyboard_arrow_right novels & short stories keyboard_arrow_right novelists l-z",
    "games & quizzes", "history & society", "science & tech", "biographies",
    "animals & nature", "geography & travel", "arts & culture",
    "ask the chatbot", "kate lohnes, patricia bauer", "britannica editors",
    "history", "account_circlekeyboard_arrow_down",
    "what is the little prince?",
    "who wrote the little prince and when was it published?",
    "who are the main characters in the little prince?",
    "what kind of journey does the little prince go on?",
    "who was antoine de saint-exupéry?",
    "what is antoine de saint-exupéry most famous for writing?",
    "where was he from, and when did he live?",
    "what are some important themes in his stories?",
}


def _strip_web_boilerplate(text: str) -> str:
    """Remove recurring website chrome (nav menus, login walls, ads, bylines,
    cross-link teasers) from study-guide/encyclopedia PDF printouts, so it
    never ends up as retrievable "content"."""
    text = _MASTHEAD_RE.sub("", text)
    text = _FOOTER_NAV_RE.sub("\n", text)
    text = _BRITANNICA_TOPNAV_RE.sub("", text)
    text = _POPULAR_PAGES_RE.sub("", text)
    text = _READ_MORE_ABOUT_RE.sub("", text)
    text = _READ_MORE_COLON_RE.sub("", text)
    text = _READ_INDEPTH_RE.sub("", text)
    text = _TRAILING_CHROME_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    return text


# True if a line looks like real prose rather than site chrome/junk
def _is_clean_line(line: str) -> bool:
    if not line:
        return False
    # bullet-separated byline/date fragments, e.g. "Kate Lohnes • All"
    if "•" in line:
        return False
    normalized = re.sub(r"\s+", " ", line.lower())
    if normalized in _BOILERPLATE_LINES or normalized in _ICON_LIGATURES:
        return False
    # drop lines that are purely numeric (page numbers)
    if re.fullmatch(r"[\d\s]+", line):
        return False
    # drop lines shorter than 4 chars
    if len(line) < 4:
        return False
    # require at least 80% of characters to be letters or spaces
    alpha = sum(c.isalpha() or c.isspace() for c in line)
    return alpha / len(line) >= 0.80


# Strip web boilerplate, then drop any remaining junk lines (nav text, page numbers, etc.)
def _clean_text(text: str) -> str:
    text = _strip_web_boilerplate(text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
        elif _is_clean_line(stripped):
            lines.append(stripped)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# The novella itself is quotable verbatim; everything else is reference material
def classify_kind(filename: str) -> str:
    return "book" if filename == BOOK_FILENAME else "reference"


# Turn a filename into a safe ID prefix for ChromaDB (letters/digits/underscores only)
def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


# Extract raw text from PDF or plain-text bytes, then strip web boilerplate.
# Shared by load_text() (disk files) and build_session_collection() (uploads).
def load_text_from_bytes(data: bytes, suffix: str) -> str:
    if suffix == ".pdf":
        doc = fitz.open(stream=data, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        text = "\n\n".join(pages)
    else:
        text = data.decode("utf-8")
    return _clean_text(text)


# Extract raw text from a PDF or plain-text file, then strip web boilerplate
def load_text(path: Path) -> str:
    return load_text_from_bytes(path.read_bytes(), path.suffix.lower())


def chunk_text(text: str, size: int, overlap: int) -> list[dict]:
    """Split text into overlapping chunks, preserving paragraph/sentence boundaries where possible.

    Paragraphs larger than `size` (e.g. a dense study-guide page with no blank
    lines) are further split at sentence boundaries so no single chunk hands
    the model an entire page verbatim.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    units = []
    for para in paragraphs:
        if len(para) <= size:
            units.append(para)
        else:
            units.extend(s for s in re.split(r"(?<=[.!?])\s+", para) if s)

    chunks = []
    current = ""

    for unit in units:
        if current and len(current) + len(unit) + 2 > size:
            chunks.append({"id": f"chunk_{len(chunks)}", "text": current.strip()})
            # keep trailing text for overlap
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + " " + unit
        else:
            current = (current + " " + unit).strip() if current else unit

    if current.strip():
        chunks.append({"id": f"chunk_{len(chunks)}", "text": current.strip()})

    return chunks


# Chunk, embed, and store every .pdf/.txt file in data_dir into a fresh ChromaDB collection
def build_index(data_dir: Path) -> None:
    paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in {".pdf", ".txt"})
    if not paths:
        print(f"Error: no .pdf or .txt files found in '{data_dir}'.")
        sys.exit(1)

    print(f"Loading embedding model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path="./chroma_db")
    # Reset existing collection so re-runs are idempotent
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total = 0
    for path in paths:
        print(f"Loading text from {path} ...")
        text = load_text(path)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            print(f"  {path.name}: no usable text, skipping.")
            continue

        kind = classify_kind(path.name)
        print(f"  {path.name}: split into {len(chunks)} chunks (kind={kind}).")

        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

        slug = slugify(path.stem)
        collection.add(
            ids=[f"{slug}_{c['id']}" for c in chunks],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[
                {"source": path.name, "kind": kind, "chunk_index": i}
                for i in range(len(chunks))
            ],
        )
        total += len(chunks)

    print(f"Indexed {total} chunks from {len(paths)} file(s) into ChromaDB at ./chroma_db")


# Build an in-memory collection from Streamlit-uploaded files — never
# written to disk, so an uploaded copyrighted book stays session-only.
def build_session_collection(book_file, reference_files: list, model: SentenceTransformer):
    client = chromadb.EphemeralClient()
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    for file, kind in [(book_file, "book"), *((f, "reference") for f in reference_files)]:
        suffix = Path(file.name).suffix.lower()
        text = load_text_from_bytes(file.read(), suffix)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            continue

        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True)

        slug = slugify(Path(file.name).stem)
        collection.add(
            ids=[f"{slug}_{c['id']}" for c in chunks],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[
                {"source": file.name, "kind": kind, "chunk_index": i}
                for i in range(len(chunks))
            ],
        )

    return collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR_DEFAULT,
        help="Directory of PDF/txt files to ingest (book + reference material)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: '{data_dir}' not found or is not a directory.")
        sys.exit(1)

    build_index(data_dir)
