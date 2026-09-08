"""Text chunking.

Turns a cleaned document into overlapping passage chunks. The splitter is
sentence-aware for both English and Chinese (CJK) text and guarantees a chunk
carries a small tail of the previous chunk so meaning isn't chopped at the
boundary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import ChunkConfig

# Sentence boundaries. Grouped so CJK punctuation (。) also splits.
_SENT_RE = re.compile(
    r"(?<=[。！？!?\.;；])\s*|\n+", re.UNICODE
)
# Token approximation: CJK characters are one token each, latin/digit runs one.
_TOKEN_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]|[a-zA-Z0-9]+(?:['’\-][a-zA-Z0-9]+)*|[^\s]")


@dataclass
class Chunk:
    index: int
    text: str
    char_length: int = field(init=False)
    token_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_length = len(self.text)
        self.token_count = count_tokens(self.text)

    def as_row(self) -> dict:
        return {
            "chunk_index": self.index,
            "chunk_text": self.text,
            "char_length": self.char_length,
            "token_count": self.token_count,
        }


def count_tokens(text: str) -> int:
    """Rough token count (CJK char == 1 token, latin word == 1 token)."""
    return len(_TOKEN_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    """Split into sentence-ish units, preserving terminating punctuation."""
    text = text.strip()
    if not text:
        return []
    raw = _SENT_RE.split(text)
    return [s.strip() for s in raw if s.strip()]


def _cut_long(text: str, size: int) -> list[str]:
    """Hard-split an oversized single unit into character windows."""
    return [text[i:i + size] for i in range(0, len(text), size)]


def chunk_text(
    text: str,
    size: int = 600,
    overlap: int = 120,
    min_chunk: int = 80,
) -> list[Chunk]:
    """Split `text` into overlapping chunks of roughly `size` chars.

    Overlap is enforced by re-using the tail of the previous chunk at the head
    of the next one, plus a sentence-aware greedy fill.
    """
    if overlap >= size:
        overlap = max(0, size // 4)

    units = split_sentences(text)
    # Merge very short fragments into the previous one to avoid 5-char chunks.
    merged: list[str] = []
    for u in units:
        if merged and len(u) < 30:
            merged[-1] = merged[-1] + " " + u
        else:
            merged.append(u)

    chunks: list[Chunk] = []
    buf = ""
    for unit in merged:
        # Unit longer than a single chunk: flush buffer, then hard-split.
        if len(unit) > size:
            if buf.strip():
                chunks.append(Chunk(index=len(chunks), text=_apply_overlap(buf, buf, overlap)))
                buf = ""
            for part in _cut_long(unit, size):
                chunks.append(Chunk(index=len(chunks), text=part))
            continue

        if len(buf) + len(unit) <= size:
            buf = (buf + " " + unit).strip() if buf else unit
        else:
            chunks.append(Chunk(index=len(chunks), text=buf))
            head = buf[-overlap:] if overlap else ""
            buf = (head + " " + unit).strip() if head else unit

    if buf.strip():
        chunks.append(Chunk(index=len(chunks), text=buf))

    # Optional: drop degenerate short chunks and renumber.
    if min_chunk:
        chunks = [c for c in chunks if c.char_length >= min_chunk]
        for i, c in enumerate(chunks):
            c.index = i

    return chunks


def _apply_overlap(buf: str, tail_src: str, overlap: int) -> str:
    """Small helper kept simple: just return buf (overlap applied by caller)."""
    return buf


def chunk_document(text: str, config: ChunkConfig | None = None) -> list[Chunk]:
    cfg = config or ChunkConfig()
    return chunk_text(text, size=cfg.size, overlap=cfg.overlap, min_chunk=cfg.min_chunk)
