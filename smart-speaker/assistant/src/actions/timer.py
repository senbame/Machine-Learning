"""Действия для установки таймеров."""
import re
import threading


NUMBER_WORDS = {
    "одну": 1, "один": 1, "два": 2, "две": 2, "три": 3,
    "четыре": 4, "пять": 5, "шесть": 6, "семь": 7,
    "восемь": 8, "девять": 9, "десять": 10,
    "пятнадцать": 15, "двадцать": 20, "тридцать": 30,
}


def parse_duration(text: str) -> int | None:
    """Разобрать длительность таймера в секундах."""
    text = text.lower().replace("ё", "е")
    if "полчаса" in text:
        return 1800
    if "полтора часа" in text:
        return 5400

    total = 0
    found = False
    pattern = r"(\d+|один|одну|одна|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять|пятнадцать|двадцать(?:\s+(?:пять|десять|пятнадцать))?|тридцать)\s*(секунд[аы]?|минут[аы]?|час(?:а|ов)?|ч)"
    for match in re.finditer(pattern, text):
        value_text, unit = match.groups()
        value_text = value_text.strip()
        if value_text.isdigit():
            value = int(value_text)
        elif " " in value_text:
            parts = value_text.split()
            value = NUMBER_WORDS.get(parts[0], 0) + sum(NUMBER_WORDS.get(part, 0) for part in parts[1:])
        else:
            value = NUMBER_WORDS.get(value_text)
        if value is None or value <= 0:
            continue
        found = True
        if unit.startswith("час") or unit == "ч":
            total += value * 3600
        elif unit.startswith("минут"):
            total += value * 60
        else:
            total += value

    return total if found and total > 0 else None


def handle(text: str, cfg: dict) -> str:
    """Установить таймер или запросить длительность."""
    seconds = parse_duration(text)
    if seconds is None:
        return "Не понял, на сколько поставить таймер"
    if cfg.get("dry_run", False):
        return f"[проверка] Таймер на {seconds} секунд"

    def on_timer() -> None:
        print("Таймер сработал")
        try:
            import sys
            if sys.platform == "win32":
                import winsound
                winsound.Beep(1000, 500)
        except Exception:
            pass

    timer = threading.Timer(seconds, on_timer)
    timer.daemon = True
    timer.start()
    return f"Таймер на {seconds} секунд установлен"
