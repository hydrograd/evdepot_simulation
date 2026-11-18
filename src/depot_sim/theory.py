"""Closed-form M/M/c results (Erlang B / Erlang C).

Notation
--------
lam : arrival rate (jobs per time unit)
mu  : service rate of one server (jobs per time unit)
c   : number of identical servers
a   : offered load lam / mu (erlangs)
rho : per-server utilisation a / c, must be < 1 for a steady state

Erlang B is computed with the standard recursion rather than the factorial
form, which overflows and loses precision for large c.
"""

from __future__ import annotations

from dataclasses import dataclass


def erlang_b(c: int, a: float) -> float:
    """Blocking probability of an M/M/c/c system: B(c, a)."""
    if c < 0 or int(c) != c:
        raise ValueError(f"c must be a non-negative integer, got {c!r}")
    if a < 0:
        raise ValueError(f"offered load must be >= 0, got {a!r}")
    b = 1.0
    for n in range(1, int(c) + 1):
        b = a * b / (n + a * b)
    return b


def _check(lam: float, mu: float, c: int) -> tuple[float, float]:
    if lam <= 0 or mu <= 0:
        raise ValueError("lam and mu must be positive")
    if c < 1 or int(c) != c:
        raise ValueError(f"c must be a positive integer, got {c!r}")
    a = lam / mu
    rho = a / c
    if rho >= 1:
        raise ValueError(f"unstable queue: rho = {rho:.4f} >= 1")
    return a, rho


def erlang_c(lam: float, mu: float, c: int) -> float:
    """Probability that an arriving job has to wait, P(Wq > 0)."""
    a, rho = _check(lam, mu, c)
    bc = erlang_b(c, a)
    return bc / (1.0 - rho * (1.0 - bc))


def mean_wait_in_queue(lam: float, mu: float, c: int) -> float:
    """Wq: mean time a job spends waiting before service starts."""
    _check(lam, mu, c)
    return erlang_c(lam, mu, c) / (c * mu - lam)


def mean_queue_length(lam: float, mu: float, c: int) -> float:
    """Lq = lam * Wq (Little's law)."""
    return lam * mean_wait_in_queue(lam, mu, c)


def mean_time_in_system(lam: float, mu: float, c: int) -> float:
    """W = Wq + 1/mu."""
    return mean_wait_in_queue(lam, mu, c) + 1.0 / mu


@dataclass(frozen=True)
class MMcResult:
    lam: float
    mu: float
    c: int
    rho: float
    p_wait: float
    Wq: float
    Lq: float
    W: float


def mmc(lam: float, mu: float, c: int) -> MMcResult:
    """All steady-state M/M/c quantities in one object."""
    a, rho = _check(lam, mu, c)
    p_wait = erlang_c(lam, mu, c)
    wq = p_wait / (c * mu - lam)
    return MMcResult(lam, mu, c, rho, p_wait, wq, lam * wq, wq + 1.0 / mu)
