"""Обучает TF-IDF + логистическую регрессию и сохраняет модель в models/.

Запуск из папки intent-classifier:
    python -m src.train                # обучение на train.csv
    python -m src.train --all          # финальная модель на train + test
"""
import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.dataset import ROOT, load_split, normalize

MODEL_PATH = ROOT / "models" / "intent_tfidf.joblib"


def build_model(C: float = 10.0):
    # normalize внутри пайплайна: модель принимает сырой текст
    return make_pipeline(
        TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2)),
        LogisticRegression(C=C, max_iter=3000),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=10.0, help="сила регуляризации (подобрана по кросс-валидации)")
    parser.add_argument("--all", action="store_true", help="обучить на train + test")
    args = parser.parse_args()

    data = load_split("train")
    if args.all:
        data = pd.concat([data, load_split("test")], ignore_index=True)

    model = build_model(args.C).fit(data["text"], data["intent"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Обучено на {len(data)} фразах, C={args.C}. Модель сохранена: {MODEL_PATH}")


if __name__ == "__main__":
    main()
