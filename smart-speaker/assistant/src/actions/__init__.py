"""Диспетчер действий ассистента."""
from typing import Callable

from src.actions.apps import handle as handle_app
from src.actions.clock import handle as handle_clock
from src.actions.timer import handle as handle_timer


HANDLERS: dict[str, Callable[[str, dict], str]] = {
    "open_app": handle_app,
    "get_time": handle_clock,
    "set_timer": handle_timer,
}


def dispatch(intent: str, text: str, cfg: dict) -> str:
    """Выполнить обработчик для интента или вернуть заглушку."""
    handler = HANDLERS.get(intent)
    if handler is None:
        return f"Команда '{intent}' распознана, но пока не реализована"
    return handler(text, cfg)
