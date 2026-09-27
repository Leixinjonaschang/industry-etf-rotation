"""统一日志：控制台 + 可选的实验目录 run.log。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT_LOGGER = "rotation"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """配置根日志器；重复调用是安全的（会替换旧 handler）。"""
    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(_FORMAT, "%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT, "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")
