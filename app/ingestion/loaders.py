from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from docx import Document


@dataclass
class LoadedDoc:
    title: str
    source: str  # filepath
    sections: List[Dict]  # each: {"text": "...", "meta": {...}}


def load_docx(path: str) -> LoadedDoc:
    p = Path(path)
    doc = Document(path)

    parts: List[str] = []
    for para in doc.paragraphs:
        txt = (para.text or "").strip()
        if txt:
            parts.append(txt)

    full_text = "\n".join(parts)

    # Single section for docx baseline; later we can split by headings
    return LoadedDoc(
        title=p.stem,
        source=str(p),
        sections=[{"text": full_text, "meta": {"type": "docx"}}],
    )
