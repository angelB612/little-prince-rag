# The Little Prince RAG

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Antoine de Saint-Exupéry's *The Little Prince*, using Gemini and ChromaDB.

![app screenshot](little_prince_fox.png)

## How it works

1. **Ingest**: every PDF/text file in `data/` is split into overlapping chunks, embedded with `all-MiniLM-L6-v2`, and stored in ChromaDB. `TheLittlePrince.pdf` is tagged `kind=book`; everything else is tagged `kind=reference`.
2. **Query**: each question is spell-corrected, embedded the same way, and the top-8 nearest passages are retrieved.
3. **Generate**: the passages go to `gemini-2.5-flash` as context. The model can quote `book` passages verbatim, but must paraphrase `reference` passages instead of reciting them.

## Requirements

- Python 3.11+
- A free [Gemini API key](https://aistudio.google.com/apikey)
- Your own copy of *The Little Prince* as a PDF or `.txt` file, named `TheLittlePrince.pdf` (or update `BOOK_FILENAME` in `ingest.py`), plus any optional reference material (background, themes, character notes, plot analysis), all placed in `data/`

`data/` and `chroma_db/` aren't in this repo. The book and any reference material are copyrighted, so bring your own copies.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY=your-key-here   # or put it in .streamlit/secrets.toml
```

Got more than one Python on your machine? Stick to the venv above. A bare `pip install` can land in a different interpreter than the one running the app.

## Usage

### 1. Ingest the book and reference material

```bash
python ingest.py
# or, to ingest a different directory
python ingest.py --data-dir mydata
```

This ingests every `.pdf`/`.txt` file found directly in `data/` (non-recursive) and creates a `chroma_db/` directory containing the vector index. Re-run it whenever you add or update a file in `data/`; it rebuilds the index from scratch.

### 2a. Launch the Streamlit UI

```bash
streamlit run app.py
```

Opens a chat interface in your browser. Retrieved source passages are shown in the sidebar after each answer.

### 2b. Use the CLI instead

```bash
# Interactive mode
python rag.py

# Single question
python rag.py "What does the fox teach the little prince?"
```

## Project structure

```
.
├── app.py          # Streamlit chat UI
├── rag.py          # Retrieval + prompt logic; CLI entry point
├── llm.py          # Chat-completion backend (Gemini; swap here for a different provider)
├── ingest.py       # Chunking, embedding, and ChromaDB indexing
├── requirements.txt
├── data/           # Source material: the novella + reference PDFs (gitignored)
└── chroma_db/      # Auto-generated vector store (gitignored, rebuild with ingest.py)
```

## Configuration

Key constants live at the top of each file:

| File | Constant | Default | Description |
|------|----------|---------|-------------|
| `llm.py` | `MODEL` | `gemini-2.5-flash` | Gemini model used for generation |
| `rag.py` | `TOP_K` | `8` | Number of passages retrieved per query |
| `rag.py` | `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer embedding model |
| `ingest.py` | `DATA_DIR_DEFAULT` | `data` | Directory scanned for `.pdf`/`.txt` files to ingest |
| `ingest.py` | `BOOK_FILENAME` | `TheLittlePrince.pdf` | The one file tagged `kind=book` (quotable verbatim); every other file is tagged `kind=reference` (paraphrase-only) |
| `ingest.py` | `CHUNK_SIZE` | `400` | Target characters per chunk |
| `ingest.py` | `CHUNK_OVERLAP` | `80` | Character overlap between adjacent chunks |
