"""Проверки ядра ассистента без модели и звука."""
from pathlib import Path

import yaml

from src.actions.apps import find_apps
from src.nlu import IntentClassifier


ROOT = Path(__file__).resolve().parent.parent


def load_apps():
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)["apps"]


def test_find_browser():
    assert find_apps("запусти бралвер", load_apps()) == ["browser"]


def test_find_youtube_removes_browser():
    assert find_apps("открой браузер и запусти ютуб", load_apps()) == ["youtube"]


def test_find_mail():
    assert find_apps("открой почту", load_apps()) == ["mail"]


def test_weather_is_not_app():
    assert find_apps("какая сегодня погода", load_apps()) == []


def test_anekdot_is_unknown():
    csv_path = ROOT.parent / "intent-classifier" / "data" / "intents.csv"
    classifier = IntentClassifier()
    classifier.fit_from_csv(csv_path, C=10)
    intent, _ = classifier.predict("расскажи анекдот")
    assert intent == "unknown"
