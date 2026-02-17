from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import fitz  # pymupdf


@dataclass
class LoadedPDF:
    title: str
    source: str
    pages: List[Dict]  # each: {"text": "...", "meta": {...}}


def load_pdf(path: str) -> LoadedPDF:
    p = Path(path)
    doc = fitz.open(path)

    pages: List[Dict] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        txt = (page.get_text("text") or "").strip()
        if not txt:
            # keep empty pages too (optional); usually skip
            continue
        pages.append(
            {
                "text": txt,
                "meta": {"type": "pdf", "page": i + 1},  # 1-based page for humans
            }
        )

    return LoadedPDF(title=p.stem, source=str(p), pages=pages)
