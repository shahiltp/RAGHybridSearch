import pandas as pd
import subprocess
from pathlib import Path

MIN_FAITHFULNESS = 0.75
MIN_ANSWER_RELEVANCY = 0.70

def test_rag_quality():
    # Run eval
    subprocess.check_call(["python", "-m", "app.eval.run_eval"])

    df = pd.read_csv("data/eval_results.csv")
    # df columns depend on ragas version; adjust if needed
    faith = float(df["faithfulness"].mean())
    rel = float(df["answer_relevancy"].mean())

    assert faith >= MIN_FAITHFULNESS, f"Faithfulness too low: {faith}"
    assert rel >= MIN_ANSWER_RELEVANCY, f"Answer relevancy too low: {rel}"
