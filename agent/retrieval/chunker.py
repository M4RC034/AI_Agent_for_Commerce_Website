"""Split the zh-TW knowledge base into self-contained retrieval chunks.

The KB documents are written with ``##`` sections that each stand alone, so the
section is the natural chunk boundary — a chunk that says "如上所述，需於七日內申請"
is useless once retrieval pulls it out of document order.

Each chunk carries its document title and section heading in the text handed to
the model, so a retrieved fragment always announces where it came from and the
agent can cite it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

#: Sections longer than this are split further on paragraph boundaries. Chosen
#: so a chunk stays well inside the reranker's input window while still holding
#: a complete policy statement.
MAX_CHARS = 600


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc: str
    title: str
    section: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def embed_text(self) -> str:
        """What actually gets embedded.

        Prefixing the document and section titles gives the bi-encoder the
        topical anchor it otherwise loses when a section body uses pronouns or
        elided subjects — common in Chinese policy prose.
        """
        return f"{self.title}｜{self.section}\n{self.text}"


def _split_long_section(body: str) -> list[str]:
    """Break an over-long section on blank lines, packing greedily."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    parts: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > MAX_CHARS and current:
            parts.append(current)
            current = para
        else:
            current = candidate

    if current:
        parts.append(current)
    return parts or [body.strip()]


def chunk_document(path: Path) -> list[Chunk]:
    """Parse one KB markdown file into chunks, one per ``##`` section."""
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    title = path.stem
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    # Split on level-2 headings, keeping the heading with its body.
    sections: list[tuple[str, str]] = []
    heading = None
    buf: list[str] = []

    for line in lines:
        if line.startswith("## "):
            if heading is not None:
                sections.append((heading, "\n".join(buf).strip()))
            heading = line[3:].strip()
            buf = []
        elif heading is not None:
            buf.append(line)

    if heading is not None:
        sections.append((heading, "\n".join(buf).strip()))

    chunks: list[Chunk] = []
    for section, body in sections:
        if not body:
            continue
        for i, part in enumerate(_split_long_section(body)):
            suffix = f"-{i}" if i else ""
            slug = re.sub(r"\W+", "-", section)[:24].strip("-")
            chunks.append(
                Chunk(
                    chunk_id=f"{path.stem}::{slug}{suffix}",
                    doc=path.name,
                    title=title,
                    section=section,
                    text=part,
                )
            )
    return chunks


def chunk_corpus(kb_dir: Path) -> list[Chunk]:
    """Chunk every ``.md`` file in the KB directory, in stable filename order."""
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        chunks.extend(chunk_document(path))
    return chunks
