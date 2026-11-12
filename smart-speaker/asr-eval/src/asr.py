"""Загрузка моделей Whisper и распознавание речи."""
import time

import librosa
import torch
from transformers import pipeline


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
    """Читает файл в моно, 16 кГц (ffmpeg не нужен)."""
    audio, _ = librosa.load(str(path), sr=16000, mono=True)
    return audio


def transcribe(asr, audio, language="russian"):
    result = asr(audio, generate_kwargs={"language": language, "task": "transcribe"})
    return result["text"].strip()