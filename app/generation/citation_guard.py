from __future__ import annotations
import re
from typing import List, Set, Tuple,Dict,Any

from app.core.types import Chunk


_CIT_RE = re.compile(r"\[([^\]:]+):([^\]]+)\]")

def extract_citations(text: str) -> Set[Tuple[str, str]]:
    return set(_CIT_RE.findall(text))

def validate_citations(answer: str, chunks: List[Chunk]) -> bool:
    allowed = {(c.doc_id, c.chunk_id) for c in chunks}
    used = extract_citations(answer)
    # Must cite at least once, and all citations must be valid
    return len(used) > 0 and used.issubset(allowed)

def safe_fallback() -> str:
    return "I don't know based on the provided documents."

def citations_with_pages(answer: str, chunks: List[Chunk]) -> List[Dict[str, Any]]:
    used = extract_citations(answer)  # set of (doc_id, chunk_id)
    chunk_map = {(c.doc_id, c.chunk_id): c for c in chunks}

    out: List[Dict[str, Any]] = []
    for doc_id, chunk_id in sorted(used):
        ch = chunk_map.get((doc_id, chunk_id))
        page = None
        if ch and isinstance(ch.metadata, dict):
            p = ch.metadata.get("page")
            if p is not None:
                try:
                    page = int(p)
                except Exception:
                    page = None
        out.append({"doc_id": doc_id, "chunk_id": chunk_id, "page": page})
    return out
