"""Загрузка моделей Whisper и распознавание речи."""
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
import librosa
import torch
from transformers import pipeline

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def load_asr(model_name: str):
    """model_name: tiny, base, small, medium... Возвращает (pipeline, время загрузки)."""
    start = time.perf_counter()
    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline(
        "automatic-speech-recognition",
        model=f"openai/whisper-{model_name}",
        device=device,
    )
    return asr, time.perf_counter() - start


def load_audio(path):
    path = Path(path)
    if path.suffix.lower() != ".wav":
        wav = path.with_suffix(".wav")
        if not wav.exists():
            subprocess.run(
                [FFMPEG, "-y", "-i", str(path), "-ar", "16000", "-ac", "1", str(wav)],
                check=True,
                capture_output=True,
            )
        path = wav

    audio, _ = librosa.load(str(path), sr=16000, mono=True)
    return audio


def transcribe(asr, audio, language="russian"):
    result = asr(audio, generate_kwargs={"language": language, "task": "transcribe"})
    return result["text"].strip()