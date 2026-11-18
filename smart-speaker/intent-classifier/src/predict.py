"""Определяет интент одной фразы.

В коде:   from src.predict import predict_intent;  predict_intent("открой ютуб")
Из консоли (папка intent-classifier):  python -m src.predict "открой ютуб"
"""
import sys

import joblib

from src.train import MODEL_PATH

_model = None


def predict_intent(text: str):
    """Возвращает (интент, уверенность). Модель загружается один раз."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    proba = _model.predict_proba([text])[0]
    i = proba.argmax()
    return _model.classes_[i], float(proba[i])


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or input("Команда: ")
    intent, conf = predict_intent(text)
    print(f"{intent} (уверенность {conf:.2f})")
