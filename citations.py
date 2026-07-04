"""
APA 7th-edition citation registry, keyed by the `source` filename stored in
each chunk's metadata (see ingest.py). Each entry gives the exact in-text
citation the model must use (embedded directly into the prompt — see
rag.py's build_prompt) and the matching full reference-list entry, so the
two can never drift apart.

The three SparkNotes pages share the same author and no date, so per APA 7
section 8.19 they're disambiguated as n.d.-a/-b/-c (ordered alphabetically
by title, matching their reference-list order).
"""

BOOK_SOURCE = "TheLittlePrince.pdf"

CITATIONS = {
    BOOK_SOURCE: {
        "in_text": "(Saint-Exupéry, 1943)",
        "reference": "Saint-Exupéry, A. de. (1943). *The little prince*. Reynal & Hitchcock.",
    },
    "Background.pdf": {
        "in_text": "(SparkNotes Editors, n.d.-a)",
        "reference": (
            "SparkNotes Editors. (n.d.-a). *The Little Prince: Background*. SparkNotes. "
            "https://www.sparknotes.com/lit/littleprince/context/"
        ),
    },
    "character-list.pdf": {
        "in_text": "(SparkNotes Editors, n.d.-b)",
        "reference": (
            "SparkNotes Editors. (n.d.-b). *The Little Prince: Character list*. SparkNotes. "
            "https://www.sparknotes.com/lit/littleprince/characters/"
        ),
    },
    "themes.pdf": {
        "in_text": "(SparkNotes Editors, n.d.-c)",
        "reference": (
            "SparkNotes Editors. (n.d.-c). *The Little Prince: Themes*. SparkNotes. "
            "https://www.sparknotes.com/lit/littleprince/themes/"
        ),
    },
    "Plot, Analysis, & Facts.pdf": {
        "in_text": "(Lohnes, 2026)",
        "reference": (
            "Lohnes, K. (2026, May 29). *The Little Prince*. Encyclopedia Britannica. "
            "https://www.britannica.com/topic/The-Little-Prince"
        ),
    },
}


# Look up the APA in-text citation; falls back to the filename itself for
# uploads that don't match a known citation.
def in_text(source: str) -> str:
    entry = CITATIONS.get(source)
    return entry["in_text"] if entry else f"({source})"


def references_cited_in(answer: str) -> list[str]:
    """Full reference-list entries for whichever in-text citations actually
    appear in the given answer text (not merely retrieved-but-unused),
    alphabetized as an APA reference list would be."""
    return sorted(c["reference"] for c in CITATIONS.values() if c["in_text"] in answer)
