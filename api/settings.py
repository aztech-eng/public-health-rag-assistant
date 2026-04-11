from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache


logger = logging.getLogger(__name__)

DEFAULT_INSUFFICIENT_ANSWER = "I don't know based on the indexed public health evidence."
DEFAULT_INSUFFICIENT_REASON = "insufficient_evidence"


@dataclass(frozen=True)
class APISettings:
    ask_min_evidence_score: float
    ask_insufficient_answer: str


@lru_cache(maxsize=1)
def get_settings() -> APISettings:
    return APISettings(
        ask_min_evidence_score=_read_env_float("ASK_MIN_EVIDENCE_SCORE", default=0.2, minimum=0.0),
        ask_insufficient_answer=_read_env_str("ASK_INSUFFICIENT_ANSWER", default=DEFAULT_INSUFFICIENT_ANSWER),
    )


def _read_env_float(name: str, *, default: float, minimum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r. Falling back to %.3f.", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s=%.3f is below %.3f. Falling back to %.3f.", name, value, minimum, default)
        return default
    return value


def _read_env_str(name: str, *, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    cleaned = raw.strip()
    if not cleaned:
        logger.warning("%s is empty after trimming. Falling back to default value.", name)
        return default
    return cleaned
