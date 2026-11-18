"""Готовит данные: train/test из intents.csv и hard_asr.csv из результатов Whisper."""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path("data")
ASR_RESULTS = Path("../asr-eval/results/results_per_file.csv")

INTENT_BY_FILE = {
    "18_37_wake_up.ogg": "set_alarm",
    "5_min_timer.ogg": "set_timer",
    "run_browser.ogg": "open_app",
    "sound_lower.ogg": "volume_down",
    "sound_upper.ogg": "volume_up",
    "weather.ogg": "get_weather",
}


def normalize(t: str) -> str:
    t = t.lower().replace("ё", "е")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    df = pd.read_csv(DATA / "intents.csv")
    train, test = train_test_split(df, test_size=0.2, stratify=df["intent"], random_state=42)
    train.to_csv(DATA / "train.csv", index=False)
    test.to_csv(DATA / "test.csv", index=False)

    asr = pd.read_csv(ASR_RESULTS)
    asr["intent"] = asr["file"].map(INTENT_BY_FILE)
    asr["text"] = asr["hypothesis"].map(normalize)
    asr[["model", "text", "intent"]].to_csv(DATA / "hard_asr.csv", index=False)

    print("train:", len(train), "| test:", len(test), "| hard_asr:", len(asr))


if __name__ == "__main__":
    main()