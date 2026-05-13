import numpy as np


def has_plateaued(history, window=6, rel_tol=0.02):
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

def has_started_increasing(history, window=6, rel_increase=0.10, grace_steps=1):
    """
    Returns True if the last `grace_steps` force values are all significantly
    worse than the best value in the preceding window.
    """
    if len(history) < window + grace_steps:
        return False

    values = np.array([h["max_force_rms"] for h in history], dtype=float)

    recent_best = values[-window-grace_steps:-grace_steps].min()
    recent_values = values[-grace_steps:]

    return np.all(recent_values > (1.0 + rel_increase) * recent_best)
