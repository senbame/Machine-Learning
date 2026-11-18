"""Оркестратор цепочки голосового ассистента."""
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from src.actions import dispatch
from src.nlu import IntentClassifier


@dataclass
class Result:
    """Результат обработки одной команды."""

    text: str
    intent: str
    confidence: float
    reply: str
    asr_s: float
    nlu_s: float
    action_s: float


class Assistant:
    """Ассистент с ленивой загрузкой Whisper."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.asr = None
        nlu_cfg = config["nlu"]
        csv_path = Path(nlu_cfg["intents_csv"])
        if not csv_path.is_absolute():
            csv_path = (Path(__file__).resolve().parent.parent / csv_path).resolve()
        self.classifier = IntentClassifier().fit_from_csv(csv_path, C=nlu_cfg["C"])

    def _ensure_asr(self) -> Any:
        if self.asr is None:
            from src.asr import load_asr

            asr_cfg = self.config["asr"]
            self.asr = load_asr(asr_cfg["model"], asr_cfg["language"])
        return self.asr

    def _process_text(self, text: str) -> Result:
        nlu_start = time.perf_counter()
        intent, confidence = self.classifier.predict(text)
        nlu_s = time.perf_counter() - nlu_start

        action_s = 0.0
        threshold = float(self.config["nlu"]["threshold"])
        if intent == "unknown" or confidence < threshold:
            reply = "Не понял, повторите"
        else:
            action_start = time.perf_counter()
            reply = dispatch(intent, text, self.config)
            action_s = time.perf_counter() - action_start

        return Result(text, intent, confidence, reply, 0.0, nlu_s, action_s)

    def handle_text(self, text: str) -> Result:
        """Обработать текстовую команду."""
        if not text.strip():
            return Result("", "unknown", 0.0, "Ничего не расслышал", 0.0, 0.0, 0.0)
        return self._process_text(text)

    def handle_audio(self, path: str) -> Result:
        """Распознать аудиофайл и обработать команду."""
        asr_start = time.perf_counter()
        from src.asr import load_audio, transcribe

        asr = self._ensure_asr()
        audio = load_audio(path)
        text = transcribe(asr, audio, self.config["asr"]["language"])
        asr_s = time.perf_counter() - asr_start
        result = self._process_text(text)
        result.asr_s = asr_s
        return result

    def handle_mic(self) -> Result:
        """Записать и обработать команду с микрофона."""
        asr_start = time.perf_counter()
        from src.asr import record, transcribe

        asr = self._ensure_asr()
        audio = record(float(self.config["asr"]["mic_seconds"]))
        text = transcribe(asr, audio, self.config["asr"]["language"])
        asr_s = time.perf_counter() - asr_start
        result = self._process_text(text)
        result.asr_s = asr_s
        return result
