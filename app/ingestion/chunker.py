from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import tiktoken


@dataclass
class ChunkedText:
    text: str
    meta: Dict


class TokenChunker:
    def __init__(self, model: str = "gpt-4o-mini", chunk_tokens: int = 350, overlap_tokens: int = 40):
        self.enc = tiktoken.encoding_for_model(model)
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text: str, base_meta: Dict) -> List[ChunkedText]:
        ids = self.enc.encode(text)
        out: List[ChunkedText] = []

        start = 0
        chunk_index = 0
        while start < len(ids):
            end = min(start + self.chunk_tokens, len(ids))
            chunk_ids = ids[start:end]
            chunk_txt = self.enc.decode(chunk_ids).strip()

            if chunk_txt:
                meta = dict(base_meta)
                meta["chunk_index"] = chunk_index
                out.append(ChunkedText(text=chunk_txt, meta=meta))

            chunk_index += 1
            if end == len(ids):
                break
            start = max(0, end - self.overlap_tokens)

        return out
