"""Проверяет сохранённую модель на test.csv и на выходах Whisper (hard_asr.csv).

Запуск из папки intent-classifier: python -m src.evaluate
"""
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.dataset import ROOT, load_split
from src.train import MODEL_PATH


def main():
    model = joblib.load(MODEL_PATH)

    test = load_split("test")
    pred = model.predict(test["text"])
    acc = accuracy_score(test["intent"], pred)
    f1 = f1_score(test["intent"], pred, average="macro")
    print(f"test: accuracy = {acc:.3f}, macro-F1 = {f1:.3f}\n")
    print(classification_report(test["intent"], pred, zero_division=0))

    hard = load_split("hard_asr")
    hard["pred"] = model.predict(hard["text"])
    hard["ok"] = hard["intent"] == hard["pred"]
    by_asr = hard.groupby("model")["ok"].mean().round(2)
    print("Доля верных интентов на выходах Whisper:")
    print(by_asr.to_string())
    print("\nОшибки:")
    print(hard.loc[~hard["ok"], ["model", "text", "intent", "pred"]].to_string(index=False))

    out = ROOT / "results" / "final_model_metrics.csv"
    out.parent.mkdir(exist_ok=True)
    row = {"test_accuracy": acc, "test_macro_f1": f1, **{f"hard_{m}": v for m, v in by_asr.items()}}
    pd.DataFrame([row]).round(3).to_csv(out, index=False)
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
