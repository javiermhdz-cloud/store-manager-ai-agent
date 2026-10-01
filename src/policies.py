"""Lexical search over the internal policy PDFs in politicas/."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import cmp_to_key, lru_cache
from pathlib import Path

from pypdf import PdfReader

from src.data_tools import SPANISH_STOPWORDS, _normalize_text, _singularize

POLICIES_DIR = Path(__file__).resolve().parents[1] / "politicas"
FAQ_DOCUMENT = "faq_gerentes_de_tienda.pdf"

# Headings are numbered ("1. Objetivo" or "1.1 Alcance") and start with an
# uppercase letter; this rejects body text that starts with a digit, such as
# dates ("12 de enero de 2026") or quantities ("5 días hábiles").
_HEADING_RE = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,2})?)\.?\s+(?P<title>[A-ZÁÉÍÓÚÑÜ].{1,78})$"
)

# The running footer carries the document version, e.g. "... · Versión 3.2".
_VERSION_RE = re.compile(r"·\s*Versión\s*(?P<version>[\w.]+)")

# These lines repeat on every page (running header/footer) and carry no content.
_BOILERPLATE_RE = re.compile(r"^H-E-B México|^Página \d+$|·\s*Versión")

# Excluded because a changelog table adds noise instead of answerable policy content.
_EXCLUDED_SECTION_SUFFIX = "Control de cambios"

BM25_K1 = 1.5
BM25_B = 0.75
TIE_BREAK_RATIO = 0.10


@dataclass(frozen=True)
class PolicyChunk:
    """A section of a policy document, with the metadata needed to cite it."""

    document: str
    section: str
    page: int
    version: str | None
    text: str


def _is_boilerplate(line: str) -> bool:
    return bool(_BOILERPLATE_RE.search(line))


def _extract_chunks(pdf_path: Path) -> list[PolicyChunk]:
    reader = PdfReader(str(pdf_path))
    chunks: list[PolicyChunk] = []
    current_section = "Introducción"
    current_page = 1
    current_version: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text and not current_section.endswith(_EXCLUDED_SECTION_SUFFIX):
            chunks.append(
                PolicyChunk(
                    document=pdf_path.name,
                    section=current_section,
                    page=current_page,
                    version=current_version,
                    text=text,
                )
            )
        buffer.clear()

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            version_match = _VERSION_RE.search(line)
            if version_match:
                current_version = version_match.group("version")
            if _is_boilerplate(line):
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


def _rank_compare(item_a: tuple[float, PolicyChunk], item_b: tuple[float, PolicyChunk]) -> int:
    score_a, chunk_a = item_a
    score_b, chunk_b = item_b
    higher_score = max(score_a, score_b)
    is_close = higher_score > 0 and (higher_score - min(score_a, score_b)) / higher_score <= TIE_BREAK_RATIO
    a_is_faq = chunk_a.document == FAQ_DOCUMENT
    b_is_faq = chunk_b.document == FAQ_DOCUMENT
    if is_close and a_is_faq != b_is_faq:
        # Within the tie-break window, the formal policy outranks the FAQ.
        return 1 if a_is_faq else -1
    if score_a != score_b:
        return -1 if score_a > score_b else 1
    if chunk_a.document != chunk_b.document:
        return -1 if chunk_a.document < chunk_b.document else 1
    if chunk_a.page != chunk_b.page:
        return -1 if chunk_a.page < chunk_b.page else 1
    return 0


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
    relevant.sort(key=cmp_to_key(_rank_compare))
    return [
        {
            "document": chunk.document,
            "section": chunk.section,
            "page": chunk.page,
            "version": chunk.version,
            "text": chunk.text,
            "score": round(score, 4),
        }
        for score, chunk in relevant[:limit]
    ]
