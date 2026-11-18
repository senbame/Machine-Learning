# Local Assistant

Локальный голосовой ассистент для Windows: аудио через Whisper, классификация интентов и выполнение разрешённых действий.

## Установка

Из папки `smart-speaker` создайте окружение и установите зависимости:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Запуск выполняйте из папки `assistant`.

## Запуск

```powershell
python main.py
python main.py --text "открой браузер" --dry-run
python main.py --audio demo\command.wav --model base --dry-run
python main.py --mic
```

Для тестов и измерений:

```powershell
python -m pytest -q
python eval\run_eval.py
```

Аргументы `--model`, `--dry-run` и `--config` переопределяют параметры `config.yaml`.

## Реализованные действия

- `open_app`: разрешённые приложения и сайты из белого списка.
- `get_time`: текущее время или дата.
- `set_timer`: таймер с фоновым потоком и звуковым сигналом Windows.

Пока не реализованы: `set_alarm`, `play_music`, `volume_up`, `volume_down`, `get_weather`. TTS отсутствует, озвучка ответов не выполняется.

## Результаты измерений

Результаты прогона аудионабора будут добавлены здесь после добавления записей в `eval/`.
