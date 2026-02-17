from __future__ import annotations
from typing import List

from app.core.types import Chunk
from app.generation.prompting import SYSTEM_PROMPT, build_user_prompt
from app.generation.citation_guard import validate_citations, safe_fallback


def _allowed_citations(chunks: List[Chunk]) -> str:
    return ", ".join([f"[{c.doc_id}:{c.chunk_id}]" for c in chunks])


class Answerer:
    def __init__(self, llm):
        self.llm = llm

    def answer(self, query: str, context_chunks: List[Chunk]) -> str:
        # Pass 1
        user_prompt = build_user_prompt(query, context_chunks)
        resp1 = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        txt1 = (resp1.text or "").strip()

        if validate_citations(txt1, context_chunks):
            return txt1

        # Pass 2 (repair): force citations using only allowed IDs
        allowed = _allowed_citations(context_chunks)
        repair_prompt = user_prompt + f"""

REPAIR INSTRUCTIONS:
- Your previous answer missed citations or used invalid ones.
- Rewrite your answer in the exact required output format:
  ANSWER: ...
  CITATIONS: ...
- Use ONLY citations from this allowed list:
  {allowed}
- If the answer is not in CONTEXT, respond exactly:
  I don't know based on the provided documents.
"""
        resp2 = self.llm.generate(SYSTEM_PROMPT, repair_prompt)
        txt2 = (resp2.text or "").strip()

        if validate_citations(txt2, context_chunks):
            return txt2

        return safe_fallback()
