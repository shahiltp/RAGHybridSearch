import json
from pathlib import Path

OUT = Path("data/golden.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Edit these 10-20 Qs to match your PDF/case study
GOLDEN = [
    {
        "id": "q1",
        "doc_id": "doc_d95f9a5dd762",
        "question": "What are the primary sources of English law?",
        "expected_keywords": ["Acts of Parliament", "cases", "courts"],
    },
    {
        "id": "q2",
        "doc_id": "doc_d95f9a5dd762",
        "question": "What does stare decisis mean in this context?",
        "expected_keywords": ["decision", "stand", "precedent"],
    },
]

with OUT.open("w", encoding="utf-8") as f:
    for row in GOLDEN:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("✅ Wrote", OUT)
