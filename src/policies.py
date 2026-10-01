"""Lexical search over the internal policy PDFs in politicas/."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader

from src.data_tools import SPANISH_STOPWORDS, _normalize_text, _singularize

POLICIES_DIR = Path(__file__).resolve().parents[1] / "politicas"

# Headings are numbered ("1. Objetivo" or "1.1 Alcance") and start with an
# uppercase letter; this rejects body text that starts with a digit, such as
# dates ("12 de enero de 2026") or quantities ("5 días hábiles").
_HEADING_RE = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,2})?)\.?\s+(?P<title>[A-ZÁÉÍÓÚÑÜ].{1,78})$"
)

# These lines repeat on every page (running header/footer) and carry no content.
_BOILERPLATE_RE = re.compile(r"^H-E-B México|^Página \d+$|·\s*Versión")

BM25_K1 = 1.5
BM25_B = 0.75


@dataclass(frozen=True)
class PolicyChunk:
    """A section of a policy document, with the metadata needed to cite it."""

    document: str
    section: str
    page: int
    text: str


def _is_boilerplate(line: str) -> bool:
    return bool(_BOILERPLATE_RE.search(line))


def _extract_chunks(pdf_path: Path) -> list[PolicyChunk]:
    reader = PdfReader(str(pdf_path))
    chunks: list[PolicyChunk] = []
    current_section = "Introducción"
    current_page = 1
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            chunks.append(
                PolicyChunk(
                    document=pdf_path.name,
                    section=current_section,
                    page=current_page,
                    text=text,
                )
            )
        buffer.clear()

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line or _is_boilerplate(line):
                continue
            match = _HEADING_RE.match(line)
            if match:
                flush()
                current_section = f"{match.group('num')} {match.group('title').strip()}"
                current_page = page_number
                continue
            buffer.append(line)
    flush()
    return chunks


@lru_cache(maxsize=1)
def _load_chunks() -> tuple[PolicyChunk, ...]:
    chunks: list[PolicyChunk] = []
    for pdf_path in sorted(POLICIES_DIR.glob("*.pdf")):
        chunks.extend(_extract_chunks(pdf_path))
    return tuple(chunks)


def _tokenize(text: str) -> list[str]:
    # Same Spanish stopword list and plural handling as src.data_tools.search_tickets.
    return [
        _singularize(token)
        for token in re.findall(r"\w+", _normalize_text(text))
        if token not in SPANISH_STOPWORDS
    ]


@lru_cache(maxsize=1)
def _build_bm25_index() -> tuple[
    tuple[PolicyChunk, ...], tuple[tuple[str, ...], ...], dict[str, int], float
]:
    chunks = _load_chunks()
    tokenized = tuple(tuple(_tokenize(chunk.text)) for chunk in chunks)
    doc_freq: dict[str, int] = {}
    for tokens in tokenized:
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    lengths = [len(tokens) for tokens in tokenized]
    avg_length = sum(lengths) / len(lengths) if lengths else 0.0
    return chunks, tokenized, doc_freq, avg_length


def _bm25_score(
    query_terms: list[str],
    tokens: tuple[str, ...],
    doc_freq: dict[str, int],
    n_docs: int,
    avg_length: float,
) -> float:
    term_counts = Counter(tokens)
    length = len(tokens)
    score = 0.0
    for term in query_terms:
        term_frequency = term_counts.get(term, 0)
        if term_frequency == 0:
            continue
        docs_with_term = doc_freq.get(term, 0)
        idf = math.log((n_docs - docs_with_term + 0.5) / (docs_with_term + 0.5) + 1)
        length_norm = (1 - BM25_B + BM25_B * (length / avg_length)) if avg_length else 1
        denominator = term_frequency + BM25_K1 * length_norm
        score += idf * (term_frequency * (BM25_K1 + 1)) / denominator
    return score


def search_policies(query: str, limit: int = 3) -> list[dict[str, object]]:
    """Rank policy chunks against a Spanish query using BM25 lexical scoring."""
    if not query.strip():
        raise ValueError("query cannot be empty")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    query_terms = _tokenize(query)
    if not query_terms:
        raise ValueError("query must contain at least one searchable word besides stopwords")

    chunks, tokenized, doc_freq, avg_length = _build_bm25_index()
    n_docs = len(chunks)
    scored = [
        (_bm25_score(query_terms, tokens, doc_freq, n_docs, avg_length), chunk)
        for chunk, tokens in zip(chunks, tokenized)
    ]
    relevant = [(score, chunk) for score, chunk in scored if score > 0]
    relevant.sort(key=lambda item: (-item[0], item[1].document, item[1].page))
    return [
        {
            "document": chunk.document,
            "section": chunk.section,
            "page": chunk.page,
            "text": chunk.text,
            "score": round(score, 4),
        }
        for score, chunk in relevant[:limit]
    ]
