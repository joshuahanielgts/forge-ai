"""Paragraph chunking and BM25. No embeddings.

Accepts .txt, .md, .csv, and .pdf. CSV is one chunk per data row, with the
header names kept in the chunk text. PDF text comes from pypdf, the only
third-party import. Any other suffix is refused.
"""
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

K1 = 1.5
B = 0.75
_TOKEN = re.compile(r"[a-z0-9]+")
_SUPPORTED = {".txt", ".md", ".csv", ".pdf"}


@dataclass
class Chunk:
    text: str
    source: str
    index: int


@dataclass
class Hit:
    score: float
    source: str
    text: str
    index: int


@dataclass
class SearchResult:
    status: str
    hits: list = field(default_factory=list)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def chunk_text(text: str, source: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Pack blank-line paragraphs. Split a paragraph only when it exceeds chunk_size.

    A packed chunk repeats up to `overlap` characters from the tail of the
    previous chunk. A paragraph longer than chunk_size is a sliding window of
    chunk_size characters, stepping by chunk_size - overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    paragraphs = _paragraphs(text)
    pieces: list[str] = []
    pending: list[str] = []

    def flush_pending():
        if not pending:
            return
        body = "\n\n".join(pending)
        if pieces and overlap:
            tail = pieces[-1][-overlap:]
            body = tail + body
        pieces.append(body)
        pending.clear()

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            flush_pending()
            pieces.extend(_windows(paragraph, chunk_size, overlap))
            continue
        candidate = paragraph if not pending else "\n\n".join(pending + [paragraph])
        if pending and len(candidate) > chunk_size:
            flush_pending()
        pending.append(paragraph)
    flush_pending()
    return [Chunk(text=piece, source=source, index=i) for i, piece in enumerate(pieces)]


def load_documents(paths, chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Load txt, md, csv, and pdf files into chunks. Refuses any other suffix."""
    chunks: list[Chunk] = []
    for path in paths:
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix not in _SUPPORTED:
            raise ValueError(f"unsupported file type: {suffix or file_path.name}")
        if suffix == ".csv":
            file_chunks = _load_csv(file_path)
        elif suffix == ".pdf":
            file_chunks = _load_pdf(file_path, chunk_size, overlap)
        else:
            text = file_path.read_text(encoding="utf-8")
            file_chunks = chunk_text(text, str(file_path), chunk_size, overlap)
        for chunk in file_chunks:
            chunks.append(Chunk(text=chunk.text, source=chunk.source, index=len(chunks)))
    return chunks


class BM25Index:
    def __init__(self, chunks, k1: float = K1, b: float = B):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self._tf = []
        df: dict[str, int] = {}
        for tokens in self._tokens:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self._tf.append(counts)
            for token in counts:
                df[token] = df.get(token, 0) + 1
        self._df = df
        self._n = len(self.chunks)
        self._avgdl = (sum(len(tokens) for tokens in self._tokens) / self._n) if self._n else 0.0

    def search(self, query: str, top_k: int = 5) -> SearchResult:
        """Rank chunks. No shared tokens, or an empty index, is status 'no_match'."""
        if not self.chunks or top_k <= 0:
            return SearchResult(status="no_match", hits=[])
        scored = []
        for i, tf in enumerate(self._tf):
            score = self._score(tokenize(query), tf, len(self._tokens[i]))
            if score > 0:
                scored.append((score, i))
        if not scored:
            return SearchResult(status="no_match", hits=[])
        scored.sort(key=lambda item: (-item[0], item[1]))
        hits = []
        for score, i in scored[:top_k]:
            chunk = self.chunks[i]
            hits.append(Hit(score=score, source=chunk.source, text=chunk.text, index=chunk.index))
        return SearchResult(status="ok", hits=hits)

    def _score(self, terms, tf, dl: int) -> float:
        if not terms:
            return 0.0
        avgdl = self._avgdl or 1.0
        total = 0.0
        for term in terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            n = self._df.get(term, 0)
            idf = math.log(1.0 + (self._n - n + 0.5) / (n + 0.5))
            denom = freq + self.k1 * (1 - self.b + self.b * dl / avgdl)
            total += idf * (freq * (self.k1 + 1)) / denom
        return total


def _paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]


def _windows(paragraph: str, size: int, overlap: int) -> list[str]:
    step = size - overlap
    windows = []
    start = 0
    while start < len(paragraph):
        windows.append(paragraph[start:start + size])
        if start + size >= len(paragraph):
            break
        start += step
    return windows


def _load_pdf(path: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise ValueError(f"could not read PDF: {path.name}: file is encrypted")
        pages = []
        for page in reader.pages:
            extracted = (page.extract_text() or "").strip()
            if extracted:
                pages.append(extracted)
    except PdfReadError as exc:
        raise ValueError(f"could not read PDF: {path.name}: {exc}") from exc
    return chunk_text("\n\n".join(pages), str(path), chunk_size, overlap)


def _load_csv(path: Path) -> list[Chunk]:
    chunks = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = " | ".join(f"{name}: {value}" for name, value in row.items())
            chunks.append(Chunk(text=text, source=str(path), index=len(chunks)))
    return chunks
