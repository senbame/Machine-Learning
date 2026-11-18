"""Точка входа локального голосового ассистента."""
import argparse
from pathlib import Path

import yaml

from src.assistant import Assistant, Result


def load_config(path: str) -> dict:
    """Загрузить YAML-конфигурацию ассистента."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def print_result(result: Result) -> None:
    """Вывести распознанный текст и метрики этапов."""
    print(f"Распознанный текст: {result.text}")
    print(f"Интент: {result.intent}")
    print(f"Уверенность: {result.confidence:.2f}")
    print(f"Ответ: {result.reply}")
    print(
        f"Время: ASR={result.asr_s:.3f} c, "
        f"NLU={result.nlu_s:.3f} c, "
        f"действие={result.action_s:.3f} c"
    )


def main() -> None:
    """Разобрать аргументы и запустить выбранный режим."""
    parser = argparse.ArgumentParser(description="Локальный голосовой ассистент")
    parser.add_argument("--text", help="обработать одну текстовую команду")
    parser.add_argument("--audio", help="путь к аудиофайлу")
    parser.add_argument("--mic", action="store_true", help="запись команд с микрофона")
    parser.add_argument("--model", choices=["tiny", "base", "small"])
    parser.add_argument("--dry-run", action="store_true", help="не запускать приложения")
    parser.add_argument("--config", default="config.yaml", help="путь к YAML-конфигурации")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["asr"]["model"] = args.model
    if args.dry_run:
        cfg["dry_run"] = True

    assistant = Assistant(cfg)

    if args.audio:
        print_result(assistant.handle_audio(args.audio))
    elif args.mic:
        try:
            while True:
                input("\nНажмите Enter и говорите (Ctrl+C для выхода)...")
                print_result(assistant.handle_mic())
        except KeyboardInterrupt:
            print("\nВыход.")
    elif args.text is not None:
        print_result(assistant.handle_text(args.text))
    else:
        print("Текстовый режим. Введите команду (пустая строка для выхода).")
        while True:
            text = input("> ").strip()
            if not text:
                break
            print_result(assistant.handle_text(text))


if __name__ == "__main__":
    main()
