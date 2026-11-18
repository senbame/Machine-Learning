"""Загрузка данных и нормализация текста."""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]      # папка intent-classifier
DATA = ROOT / "data"


def normalize(text: str) -> str:
    """Нижний регистр, без знаков препинания и лишних пробелов."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_split(name: str) -> pd.DataFrame:
    """name: 'train', 'test' или 'hard_asr'."""
    return pd.read_csv(DATA / f"{name}.csv")
