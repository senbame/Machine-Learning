"""Измерить качество цепочки ассистента по набору аудиозаписей."""
import csv
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.actions.apps import find_apps
from src.assistant import Assistant


ASSISTANT_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent


def load_config() -> dict:
    """Загрузить конфигурацию ассистента для прогона."""
    with (ASSISTANT_DIR / "config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["dry_run"] = True
    return config


def action_is_correct(result, expected_action: str, config: dict) -> bool:
    """Проверить действие по распознанному тексту или отсутствию действия."""
    if not expected_action:
        return True
    return expected_action in find_apps(result.text, config["apps"])


def main() -> None:
    """Прогнать шаблон команд и сохранить результаты."""
    config = load_config()
    assistant = Assistant(config)
    input_path = EVAL_DIR / "commands.csv"
    output_path = EVAL_DIR / "results.csv"

    rows = []
    for item in csv.DictReader(input_path.open("r", encoding="utf-8")):
        audio_path = item["audio_path"]
        resolved = (ASSISTANT_DIR / audio_path).resolve()
        if not resolved.exists():
            print(f"Пропущено: файл отсутствует — {audio_path}")
            continue
        result = assistant.handle_audio(str(resolved))
        rows.append({
            "audio_path": audio_path,
            "text": result.text,
            "intent": result.intent,
            "confidence": result.confidence,
            "intent_ok": result.intent == item["expected_intent"],
            "action_ok": action_is_correct(result, item["expected_action"].strip(), config),
            "asr_s": result.asr_s,
            "nlu_s": result.nlu_s,
            "action_s": result.action_s,
        })

    fields = [
        "audio_path", "text", "intent", "confidence", "intent_ok", "action_ok",
        "asr_s", "nlu_s", "action_s",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if not rows:
        print("Нет доступных аудиофайлов для измерения.")
        return

    intent_ok = sum(bool(row["intent_ok"]) for row in rows) / len(rows)
    action_ok = sum(bool(row["action_ok"]) for row in rows) / len(rows)
    asr_median = statistics.median(row["asr_s"] for row in rows)
    nlu_median = statistics.median(row["nlu_s"] for row in rows)
    action_median = statistics.median(row["action_s"] for row in rows)
    total_median = statistics.median(
        row["asr_s"] + row["nlu_s"] + row["action_s"] for row in rows
    )

    print(f"Доля верных интентов: {intent_ok:.1%}")
    print(f"Доля верных действий: {action_ok:.1%}")
    print(f"Медианное ASR: {asr_median:.3f} c")
    print(f"Медианное NLU: {nlu_median:.3f} c")
    print(f"Медианное действие: {action_median:.3f} c")
    print(f"Медианное общее время: {total_median:.3f} c")
    print(f"Результаты сохранены в {output_path}")


if __name__ == "__main__":
    main()
