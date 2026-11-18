"""Действия для получения времени и даты."""
from datetime import datetime


MONTHS_GENITIVE = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]


def handle(text: str, cfg: dict) -> str:
    """Вернуть текущее время или дату по словам команды."""
    now = datetime.now()
    if any(word in text.lower() for word in ("числ", "дата", "день")):
        month = MONTHS_GENITIVE[now.month - 1]
        return f"Сегодня {now.day} {month} {now.year} года"
    return f"Сейчас {now:%H:%M}"
