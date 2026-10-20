"""Friendly error messages. Tracebacks go to the log file, never to the UI."""
from __future__ import annotations

import logging
import traceback
from typing import Any

log = logging.getLogger("sightline")

KNOWN: dict[str, tuple[str, str]] = {
    "no_dataset": ("Load a dataset first.", "Сначала загрузите данные."),
    "not_trained": ("Train the model first, then pick a point.", "Сначала обучите модель, потом выберите точку."),
    "unknown_point": ("That point no longer exists. Pick another one.", "Этой точки больше нет. Выберите другую."),
    "incompatible": ("This data doesn't fit the model. See the suggested fixes.", "Эти данные не подходят модели. Посмотрите варианты исправления."),
    "bad_csv": ("Couldn't read this file as a table. Is it a CSV with a header row?",
                "Не удалось прочитать файл как таблицу. Это CSV со строкой заголовков?"),
    "duplicate_columns": ("Two columns have the same name. Rename one of them.", "У двух столбцов одинаковое имя. Переименуйте один."),
    "engine_restarting": ("The engine is restarting after a cancelled fit. One moment.",
                          "Движок перезапускается после отмены. Секунду."),
    "worker_crashed": ("The training process stopped unexpectedly. It has been restarted — try again.",
                       "Процесс обучения неожиданно остановился. Он перезапущен — попробуйте ещё раз."),
    "bad_message": ("The app sent a message the engine didn't understand.", "Приложение отправило непонятное движку сообщение."),
}


def friendly(code: str, exc: BaseException | None = None, rid: int | None = None) -> dict[str, Any]:
    if exc is not None:
        log.error("%s: %s\n%s", code, exc, "".join(traceback.format_exception(exc)))
    if code in KNOWN:
        en, ru = KNOWN[code]
    elif isinstance(exc, ValueError):
        first = str(exc).strip().splitlines()[0][:300] if str(exc).strip() else ""
        en = f"scikit-learn rejected this setting: {first}" if first else "scikit-learn rejected this setting."
        ru = f"scikit-learn не принял эту настройку: {first}" if first else "scikit-learn не принял эту настройку."
        code = "invalid_params"
    elif isinstance(exc, MemoryError):
        en, ru = ("Not enough memory for this model and data size. Try fewer rows.",
                  "Не хватает памяти для этой модели и объёма данных. Попробуйте меньше строк.")
        code = "memory"
    else:
        en, ru = ("Something went wrong inside the engine. Details were written to the log file.",
                  "Внутри движка что-то пошло не так. Подробности записаны в журнал.")
    return {"type": "error", "code": code, "message": {"en": en, "ru": ru}, "rid": rid}
