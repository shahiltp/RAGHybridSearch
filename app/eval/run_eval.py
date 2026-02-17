import json
import requests
from pathlib import Path
import pandas as pd

from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate

GOLDEN_PATH = Path("data/golden.jsonl")
API_URL = "http://127.0.0.1:8000/ask"


def load_golden():
    rows = []
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def call_ask(question: str, doc_id: str):
    payload = {"query": question, "doc_id": doc_id, "debug": True}
    r = requests.post(API_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    rows = load_golden()

    eval_rows = []
    for r in rows:
        out = call_ask(r["question"], r["doc_id"])

        # Build contexts list from debug fused_top (or reranked/context_pages if you prefer)
        contexts = []
        if out.get("debug") and out["debug"].get("fused_top"):
            for item in out["debug"]["fused_top"]:
                contexts.append(item["preview"])

        eval_rows.append(
            {
                "question": r["question"],
                "answer": out["answer"],
                "contexts": contexts,
                # optional "ground_truth" if you have it (otherwise skip metrics needing it)
                # "ground_truth": r.get("ground_truth", "")
            }
        )

    ds = Dataset.from_list(eval_rows)

    result = evaluate(
        ds,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    df = result.to_pandas()
    print("\n=== RAGAS RESULTS ===")
    print(df)

    out_csv = Path("data/eval_results.csv")
    df.to_csv(out_csv, index=False)
    print("\n✅ Saved:", out_csv)


if __name__ == "__main__":
    main()
