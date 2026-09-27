from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

_LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}
_MIN_LEVEL = _LEVELS.get(os.getenv("LOG_LEVEL", "info").strip().lower(), 20)


def _render(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value).replace(" ", "_")


class StageLogger:
    def __init__(self, stage: str) -> None:
        self.stage = stage

    def _emit(self, level: str, message: str | None, fields: dict) -> None:
        if _LEVELS[level] < _MIN_LEVEL:
            return
        parts = [f"stage={self.stage}"]
        if message:
            parts.append(message)
        parts.extend(f"{key}={_render(value)}" for key, value in fields.items())
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"{stamp} {level.upper():<5} " + " ".join(parts), file=sys.stdout)

    def debug(self, message: str | None = None, **fields: object) -> None:
        self._emit("debug", message, fields)

    def info(self, message: str | None = None, **fields: object) -> None:
        self._emit("info", message, fields)

    def warn(self, message: str | None = None, **fields: object) -> None:
        self._emit("warn", message, fields)

    def error(self, message: str | None = None, **fields: object) -> None:
        self._emit("error", message, fields)


def get_logger(stage: str) -> StageLogger:
    return StageLogger(stage)
