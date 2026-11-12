"""Нормализация текста и метрики качества."""
import re

import jiwer


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)  # убрать пунктуацию и двоеточия
    return re.sub(r"\s+", " ", text).strip()


def wer(reference: str, hypothesis: str) -> float:
    return jiwer.wer(normalize(reference), normalize(hypothesis))


def cer(reference: str, hypothesis: str) -> float:
    return jiwer.cer(normalize(reference), normalize(hypothesis))