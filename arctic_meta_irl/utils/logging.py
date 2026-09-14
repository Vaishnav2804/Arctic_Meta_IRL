"""Console + (optional) TensorBoard logging."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "arctic_meta_irl", level: str = "INFO") -> logging.Logger:
    """Return a stdout logger; a handler is attached only once per name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


class TBWriter:
    """Thin TensorBoard wrapper; no-op if tensorboard is unavailable."""

    def __init__(self, log_dir: str | Path, enabled: bool = True):
        self._w = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                self._w = SummaryWriter(str(log_dir))
            except Exception:
                self._w = None

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self._w is not None:
            self._w.add_scalar(tag, value, step)

    def close(self) -> None:
        if self._w is not None:
            self._w.close()
