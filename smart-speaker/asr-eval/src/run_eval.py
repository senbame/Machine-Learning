"""Прогон моделей Whisper по записям, сохранение метрик в results/."""
import argparse
import gc
import time
from pathlib import Path

import jiwer
import pandas as pd
import torch

from .asr import load_asr, load_audio, transcribe
from .metrics import cer, normalize, wer

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "data" / "my_recordings"
REFS = ROOT / "data" / "references.csv"
RESULTS = ROOT / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["tiny", "base", "small"])
    parser.add_argument("--language", default="russian")
    args = parser.parse_args()

    refs = pd.read_csv(REFS)
    RESULTS.mkdir(exist_ok=True)

    rows, load_times = [], {}
    for model in args.models:
        print(f"\n=== whisper-{model} ===")
        asr, load_time = load_asr(model)
        load_times[model] = load_time

        # прогрев: первый вызов всегда медленнее
        transcribe(asr, load_audio(AUDIO_DIR / refs.iloc[0]["file"]), args.language)

        for _, r in refs.iterrows():
            audio = load_audio(AUDIO_DIR / r["file"])
            duration = len(audio) / 16000
            times = []
            for _ in range(3):  # прогрев: несколько первых вызовов всегда медленнее
                start = time.perf_counter()
                hyp = transcribe(asr, audio, args.language)
                times.append(time.perf_counter() - start)
            elapsed = time.perf_counter() - start

            rows.append({
                "model": model,
                "file": r["file"],
                "reference": r["text"],
                "hypothesis": hyp,
                "wer": wer(r["text"], hyp),
                "cer": cer(r["text"], hyp),
                "time_s": elapsed,
                "audio_s": duration,
                "rtf": elapsed / duration,
            })
            print(f"{r['file']:22} | {elapsed:5.2f} с | {hyp}")

        del asr
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "results_per_file.csv", index=False, encoding="utf-8-sig")

    summary = []
    for model, g in df.groupby("model", sort=False):
        refs_n = [normalize(x) for x in g["reference"]]
        hyps_n = [normalize(x) for x in g["hypothesis"]]
        summary.append({
            "model": model,
            "WER": jiwer.wer(refs_n, hyps_n),
            "CER": jiwer.cer(refs_n, hyps_n),
            "median_time_s": g["time_s"].median(),
            "load_time_s": load_times[model],
        })
    summary = pd.DataFrame(summary).round(3)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()