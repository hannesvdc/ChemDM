import numpy as np


def has_plateaued(history, window=50, rel_tol=0.02):
    """
    Returns True if the best max_force_rms over the recent window
    improved by less than rel_tol compared to the best before the window.
    """
    if len(history) < window + 1:
        return False
    values = np.array([h["max_force_rms"] for h in history], dtype=float)

    previous_best = values[:-window].min()
    recent_best = values[-window:].min()

    improvement = (previous_best - recent_best) / max(abs(previous_best), 1e-12)
    return improvement < rel_tol

def has_started_increasing(history, window=20, rel_increase=0.10):
    """
    Returns True if the current force is significantly worse than
    the recent best.
    """
    if len(history) < window:
        return False
    values = np.array([h["max_force_rms"] for h in history], dtype=float)

    min_index = len(values) - window
    recent_best = values[min_index:].min()

    current = values[-1]
    return current > (1.0 + rel_increase) * recent_best

