"""Обработчики разрешённых приложений и сайтов."""
import difflib
import os
import subprocess
import sys
import webbrowser


def find_apps(text: str, apps_cfg: dict, cutoff: float = 0.70) -> list[str]:
    """Найти приложения в тексте по белому списку."""
    found: list[str] = []
    for word in text.split():
        best_key: str | None = None
        best_score = 0.0
        for key, app in apps_cfg.items():
            for alias in app["aliases"]:
                score = difflib.SequenceMatcher(None, word, alias).ratio()
                if score > best_score:
                    best_key, best_score = key, score
        if best_score >= cutoff and best_key is not None and best_key not in found:
            found.append(best_key)
    if "youtube" in found and "browser" in found:
        found.remove("browser")
    return found


def run_app(key: str, apps_cfg: dict, dry_run: bool) -> str:
    """Запустить одно приложение из белого списка."""
    app = apps_cfg[key]
    say = app["say"]
    kind = app["kind"]
    target = app["target"]
    if dry_run:
        return f"[проверка] {say} ({kind}: {target})"
    if kind == "url":
        webbrowser.open(target)
    elif kind == "cmd":
        if sys.platform != "win32":
            return "Эта команда работает только в Windows"
        subprocess.Popen([target])
    elif kind == "uri":
        if sys.platform != "win32":
            return "Эта команда работает только в Windows"
        os.startfile(target)
    return say


def handle(text: str, cfg: dict) -> str:
    """Обработать запрос открытию приложения."""
    apps = find_apps(text, cfg["apps"])
    if not apps:
        return "Не понял, что открыть"
    return "; ".join(run_app(key, cfg["apps"], cfg.get("dry_run", False)) for key in apps)
