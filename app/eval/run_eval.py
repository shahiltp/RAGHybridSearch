import json
import requests
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

GOLDEN_PATH = Path("data/golden.jsonl")
API_URL = "http://127.0.0.1:8000/ask"


def load_golden():
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(f"{GOLDEN_PATH} not found. Create it first.")
    rows = []
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def call_ask(question: str, doc_id: str):
    payload = {"query": question, "doc_id": doc_id, "debug": True}
    r = requests.post(API_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def main():
    rows = load_golden()

    eval_rows = []
    for r in rows:
        out = call_ask(r["question"], r["doc_id"])

        # Prefer full contexts from API debug (you should add debug["contexts"] = [c.text ...])
        contexts = []
        if out.get("debug") and out["debug"].get("contexts"):
            contexts = out["debug"]["contexts"]
        elif out.get("debug") and out["debug"].get("fused_top"):
            contexts = [x.get("preview", "") for x in out["debug"]["fused_top"]]

        eval_rows.append(
            {
                "question": r["question"],
                "answer": out["answer"],
                "contexts": contexts,
                "reference": r["reference"],
            }
        )

    ds = Dataset.from_list(eval_rows)

    result = evaluate(
        ds,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
    )

    df = result.to_pandas()
    print("\n=== RAGAS RESULTS ===")
    print(df)

    out_csv = Path("data/eval_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("\n✅ Saved:", out_csv)


if __name__ == "__main__":
    main()
