"""Configuracion centralizada de logging.

Escribe a consola y, opcionalmente, a un fichero rotativo. Todos los modulos
del bot obtienen su logger con :func:`get_logger`.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _ColorFormatter(logging.Formatter):
    """Formatter con color ANSI cuando la salida es un TTY."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;41m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        color = self.COLORS.get(record.levelname)
        if color:
            # Coloreamos solo la etiqueta de nivel para no romper el grep.
            text = text.replace(record.levelname, f"{color}{record.levelname}{self.RESET}", 1)
        return text


def setup_logging(
    level: str = "INFO",
    log_file: str | os.PathLike[str] | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Configura el logging raiz. Idempotente: solo actua la primera vez."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stdout)
    use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    formatter_cls = _ColorFormatter if use_color else logging.Formatter
    stream.setFormatter(formatter_cls(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(stream)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(file_handler)

    # Las librerias de terceros son muy ruidosas en DEBUG.
    for noisy in ("urllib3", "ccxt", "yfinance", "peewee", "matplotlib", "optuna"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger ya configurado."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
