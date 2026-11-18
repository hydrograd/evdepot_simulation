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


def summarise_depot(cfg, sessions, departures, trips, tracker: BusyTracker, peak_kw: float) -> dict:
    """Per-run depot metrics, measured after the warm-up.

    A session counts if the van returned on or after day ``warmup_days``. A
    departure counts if it follows such a session (day >= warmup_days + 1),
    so every counted night has exactly one counted departure per van.
    Waits are in hours, from return to the start of charging; a van that never
    started charging is counted with its full wait until departure.
    """
    s = [r for r in sessions if r["day"] >= cfg.warmup_days]
    d = [r for r in departures if r["day"] >= cfg.warmup_days + 1]
    t = [r for r in trips if r["day"] >= cfg.warmup_days]
    nights = cfg.n_days - cfg.warmup_days

    wait = np.array([r["wait"] for r in s])
    dur = np.array([r["duration"] for r in s if r["got_power"]])
    util = tracker.utilisation()
    return {
        "n_sessions": len(s),
        "mean_wait": float(wait.mean()),
        "p95_wait": float(np.percentile(wait, 95)),
        "mean_charge_duration": float(dur.mean()) if dur.size else 0.0,
        "missed_charge": float(np.mean([not r["got_power"] for r in s])),
        "on_time": float(np.mean([r["ready"] for r in d])),
        "utilisation": float(util.mean()),
        "charger_utilisation": util.tolist(),
        "peak_power_kw": float(peak_kw),
        "energy_per_day_kwh": float(sum(r["energy_kwh"] for r in s) / nights),
        "stranded_trips": int(sum(r["stranded"] for r in t)),
    }
