from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass
class LLMResponse:
    text: str


class OpenAILLM:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return LLMResponse(text=resp.choices[0].message.content or "")
