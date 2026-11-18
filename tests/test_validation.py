"""The validation gate: the SimPy engine against Erlang C.

Validation mode is a textbook M/M/c queue, so the analytical mean wait in
queue (Wq) is the ground truth. Two assertions per load point:

1. the simulated mean is within 5% of Wq, and
2. Wq lies inside the 95% confidence interval of the simulated mean
   (the stronger statement).

Seeds are fixed (0..n-1), so the outcome is deterministic. Note that a 95%
interval is expected to miss the true value 5% of the time; across four
independent points that is about a 19% chance that at least one misses for an
arbitrary seed set. A miss by a hair is not a bug; a miss by many half-widths is.

The full-length grid is marked ``slow``. The fast version uses a shorter
horizon so the default suite runs in seconds.
"""

import os

import pytest

from depot_sim.config import ValidationConfig
from depot_sim.runner import confidence_interval, run_replications, summarise
from depot_sim.theory import mmc

GRID = [(c, rho) for c in (3, 5) for rho in (0.6, 0.8)]
N_JOBS = min(4, os.cpu_count() or 1)


def _check(c, rho, n_runs, ci_slack=1.0, **horizon):
    """``ci_slack`` widens the coverage check: 1.0 = the 95% CI itself."""
    cfg = ValidationConfig.from_rho(rho, c, **horizon)
    runs = run_replications(cfg, n_runs=n_runs, n_jobs=N_JOBS)
    theory = mmc(cfg.lam, cfg.mu, c)
    ci = summarise(runs, "mean_wait")
    rel_err = abs(ci.mean - theory.Wq) / theory.Wq
    assert rel_err < 0.05, f"c={c} rho={rho}: sim {ci} vs Erlang C {theory.Wq:.4f}"
    assert abs(ci.mean - theory.Wq) <= ci_slack * ci.half_width, (
        f"c={c} rho={rho}: Erlang C {theory.Wq:.4f} more than "
        f"{ci_slack} half-widths from {ci}"
    )
    return runs, theory


@pytest.mark.parametrize("c,rho", GRID)
def test_fast_mean_wait_matches_erlang_c(c, rho):
    # Short runs, 10 replications: a smoke test, so coverage is checked at
    # 2 half-widths. The strict 95% coverage check is the slow test below.
    _check(c, rho, n_runs=10, ci_slack=2.0, horizon=20_000.0, warmup=2_000.0)


@pytest.mark.slow
@pytest.mark.parametrize("c,rho", GRID)
def test_full_mean_wait_matches_erlang_c(c, rho):
    runs, theory = _check(c, rho, n_runs=30, horizon=200_000.0, warmup=20_000.0)
    # The interval must be tight enough to mean something.
    assert summarise(runs, "mean_wait").rel_half_width < 0.05


@pytest.mark.parametrize("c,rho", [(3, 0.8), (5, 0.6)])
def test_p_wait_and_utilisation_match_theory(c, rho):
    cfg = ValidationConfig.from_rho(rho, c, horizon=20_000.0, warmup=2_000.0)
    runs = run_replications(cfg, n_runs=10, n_jobs=N_JOBS)
    theory = mmc(cfg.lam, cfg.mu, c)
    assert summarise(runs, "p_wait").mean == pytest.approx(theory.p_wait, rel=0.03)
    assert summarise(runs, "utilisation").mean == pytest.approx(rho, rel=0.01)


def test_measures_queue_wait_not_time_in_system():
    cfg = ValidationConfig.from_rho(0.6, 3, horizon=20_000.0, warmup=2_000.0)
    runs = run_replications(cfg, n_runs=5)
    theory = mmc(cfg.lam, cfg.mu, 3)
    mean = summarise(runs, "mean_wait").mean
    assert abs(mean - theory.Wq) < abs(mean - theory.W)


def test_same_seed_same_result_and_runs_differ():
    cfg = ValidationConfig.from_rho(0.7, 3, horizon=2_000.0, warmup=200.0)
    a = run_replications(cfg, n_runs=3)
    b = run_replications(cfg, n_runs=3, n_jobs=2)
    assert [r["mean_wait"] for r in a] == [r["mean_wait"] for r in b]
    assert len({r["mean_wait"] for r in a}) == 3


def test_ci_uses_student_t():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = confidence_interval(x)
    # t(0.975, 4) = 2.7764; s = 1.5811; se = 0.7071
    assert ci.half_width == pytest.approx(2.7764 * 1.5811 / 5**0.5, rel=1e-3)
