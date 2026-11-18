"""Распознавание речи и загрузка аудио."""
from pathlib import Path
import subprocess

import imageio_ffmpeg
import librosa
import numpy as np
import torch
from transformers import pipeline


FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SAMPLE_RATE = 16000


def load_asr(model_name: str, language: str):
    """Загрузить Whisper-модель для распознавания речи."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "automatic-speech-recognition",
        model=f"openai/whisper-{model_name}",
        device=device,
    )


def load_audio(path: str | Path) -> np.ndarray:
    """Загрузить аудиофайл как моно-массив с частотой 16 кГц."""
    path = Path(path)
    if path.suffix.lower() != ".wav":
        wav_path = path.with_suffix(".wav")
        if not wav_path.exists():
            subprocess.run(
                [FFMPEG, "-y", "-i", str(path), "-ar", str(SAMPLE_RATE), "-ac", "1", str(wav_path)],
                check=True,
                capture_output=True,
            )
        path = wav_path
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio


def record(seconds: float) -> np.ndarray:
    """Записать звук с микрофона в течение заданного числа секунд."""
    import sounddevice as sd

    print(f"Говорите ({seconds:.0f} с)...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def transcribe(asr, audio: np.ndarray, language: str) -> str:
    """Распознать аудио как русский текст."""
    result = asr(audio, generate_kwargs={"language": language, "task": "transcribe"})
    return result["text"].strip()
