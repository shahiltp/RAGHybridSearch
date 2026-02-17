from __future__ import annotations
from typing import List

from app.core.types import Chunk


SYSTEM_PROMPT = """You are a Legal Document Assistant.

You MUST follow these rules:
1) Use ONLY the provided CONTEXT. Do not use outside knowledge.
2) Every answer MUST include citations in the exact format: [doc_id:chunk_id]
3) Use ONLY citations from the provided CONTEXT chunk list.
4) If the CONTEXT does not contain the answer, reply exactly:
   I don't know based on the provided documents.

Output format (exactly):
ANSWER: <your answer>
CITATIONS: [doc_id:chunk_id], [doc_id:chunk_id]
"""


def build_user_prompt(query: str, chunks: List[Chunk]) -> str:
    ctx_lines = []
    for i, ch in enumerate(chunks, start=1):
        ctx_lines.append(
            f"({i}) doc_id={ch.doc_id} chunk_id={ch.chunk_id}\n{ch.text}"
        )
    context_block = "\n\n".join(ctx_lines)

    return f"""QUESTION:
{query}

CONTEXT:
{context_block}

INSTRUCTIONS:
- Answer the QUESTION using only the CONTEXT.
- Add citations like [doc_id:chunk_id] for every claim.
- If not enough information, say "I don't know based on the provided documents."
"""
