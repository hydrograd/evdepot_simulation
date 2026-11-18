"""Independent replications and confidence intervals.

Each replication gets its own ``numpy.random.Generator`` seeded with its index
(0..n-1), passed explicitly into the model. Nothing reads global random state,
so replications are independent and every run is reproducible.

The confidence interval is built from the n per-run means using Student's t
with n - 1 degrees of freedom (not the normal 1.96).
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .config import DepotConfig, ValidationConfig

Config = ValidationConfig | DepotConfig


def run_one(cfg: Config, seed: int) -> dict:
    """One replication with its own generator."""
    from . import model  # local import keeps worker start-up cheap

    rng = np.random.default_rng(seed)
    if cfg.mode == "validation":
        out = model.run_validation(cfg, rng)
    elif cfg.mode == "depot":
        out = model.run_depot(cfg, rng)
    else:  # pragma: no cover
        raise ValueError(f"unknown mode {cfg.mode!r}")
    out["seed"] = seed
    return out


def _run_seed(args: tuple[Config, int]) -> dict:
    return run_one(*args)


def run_replications(
    cfg: Config, n_runs: int = 30, first_seed: int = 0, n_jobs: int = 1
) -> list[dict]:
    """Run ``n_runs`` replications with seeds first_seed..first_seed + n_runs - 1.

    ``n_jobs > 1`` spreads runs over processes; results are identical either way
    because each run depends only on its own seed.
    """
    seeds = range(first_seed, first_seed + n_runs)
    if n_jobs == 1:
        return [run_one(cfg, s) for s in seeds]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(_run_seed, [(cfg, s) for s in seeds]))


@dataclass(frozen=True)
class CI:
    mean: float
    half_width: float
    n: int
    level: float = 0.95

    @property
    def low(self) -> float:
        return self.mean - self.half_width

    @property
    def high(self) -> float:
        return self.mean + self.half_width

    @property
    def rel_half_width(self) -> float:
        """Half-width as a fraction of the mean."""
        return self.half_width / abs(self.mean) if self.mean else math.inf

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def __str__(self) -> str:
        return (
            f"{self.mean:.4f} ± {self.half_width:.4f} "
            f"({100 * self.rel_half_width:.1f}%, {self.level:.0%} CI, n={self.n})"
        )


def confidence_interval(x, level: float = 0.95) -> CI:
    """t-based CI for the mean of independent observations ``x``."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("need at least two replications for a confidence interval")
    se = x.std(ddof=1) / math.sqrt(n)
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    return CI(float(x.mean()), float(t * se), n, level)


def summarise(runs: list[dict], key: str, level: float = 0.95) -> CI:
    """CI across runs for one per-run metric."""
    return confidence_interval([r[key] for r in runs], level)
