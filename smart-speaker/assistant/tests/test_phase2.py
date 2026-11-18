"""Проверки дополнительных действий фазы 2."""
from src.actions.timer import parse_duration


def test_timer_five_minutes():
    assert parse_duration("таймер на 5 минут") == 300


def test_timer_half_hour():
    assert parse_duration("таймер на полчаса") == 1800


def test_timer_composite():
    assert parse_duration("1 час 15 минут") == 4500


def test_timer_missing_duration():
    assert parse_duration("поставь таймер") is None
