"""Validation of the numeric command arguments every physical tool call carries."""
import numpy as np

MAX_CHUNK_TICKS = 200


def finite_vector(value, length, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} requires {length} finite numbers")
    return result


def bounded_ticks(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= value <= MAX_CHUNK_TICKS:
        raise ValueError(f"ticks must be an integer in [1, {MAX_CHUNK_TICKS}]")
    return int(value)

