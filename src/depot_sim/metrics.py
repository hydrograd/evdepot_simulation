"""Per-run bookkeeping and summaries.

The summaries return one dict per run. Across runs, the runner treats each
run's mean as one observation: jobs inside a run are correlated, so pooling
every job from every run into one mean would understate the uncertainty.
"""

from __future__ import annotations

import numpy as np


class BusyTracker:
    """Time each charger is held, clipped to an observation window."""

    def __init__(self, n: int, window_start: float, window_end: float) -> None:
        self.busy = np.zeros(n)
        self.ws = window_start
        self.we = window_end

    def add(self, charger_id: int, start: float, end: float) -> None:
        overlap = min(end, self.we) - max(start, self.ws)
        if overlap > 0:
            self.busy[charger_id] += overlap

    def utilisation(self) -> np.ndarray:
        return self.busy / (self.we - self.ws)


def summarise_validation(waits: list[float], tracker: BusyTracker) -> dict:
    w = np.asarray(waits)
    util = tracker.utilisation()
    return {
        "n_jobs": int(w.size),
        "mean_wait": float(w.mean()),
        "p_wait": float((w > 0).mean()),
        "utilisation": float(util.mean()),
        "charger_utilisation": util.tolist(),
    }
