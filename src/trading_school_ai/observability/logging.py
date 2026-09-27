"""Structured logging with secret masking."""
from __future__ import annotations

import logging
import os
import re
import sys

SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSPHRASE", "PASSWORD")
_MASK_RE = re.compile(r"(sk-[A-Za-z0-9]{6,}|Bearer\s+[A-Za-z0-9\-\._]{8,})")


class MaskingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        masked = _MASK_RE.sub("***REDACTED***", msg)
        for name, value in os.environ.items():
            if value and len(value) > 6 and any(h in name.upper() for h in SECRET_ENV_HINTS):
                masked = masked.replace(value, "***REDACTED***")
        record.msg = masked
        record.args = ()
        return True


def get_logger(name: str = "tsa", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        handler.addFilter(MaskingFilter())
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
