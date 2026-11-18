"""Классификация текстовых команд ассистента."""
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def normalize(text: str) -> str:
    """Нормализовать текст команды для классификации."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class IntentClassifier:
    """Классификатор интентов на TF-IDF и логистической регрессии."""

    def __init__(self) -> None:
        self.model = None

    def fit_from_csv(self, path: str | Path, C: float) -> "IntentClassifier":
        """Обучить модель по CSV с колонками text и intent."""
        df = pd.read_csv(path)
        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2)),
            LogisticRegression(C=C, max_iter=3000),
        )
        self.model.fit(df["text"].map(normalize), df["intent"])
        return self

    def predict(self, text: str) -> tuple[str, float]:
        """Вернуть наиболее вероятный интент и его уверенность."""
        if self.model is None:
            raise RuntimeError("Модель интентов ещё не обучена")
        probabilities = self.model.predict_proba([normalize(text)])[0]
        index = int(probabilities.argmax())
        return str(self.model.classes_[index]), float(probabilities[index])
