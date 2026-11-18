"""All model parameters and their defaults, in one place.

Two configurations drive the same charger/queue machinery:

* ``ValidationConfig`` - an M/M/c queue (Poisson arrivals, exponential service,
  c identical servers, FCFS, unlimited queue). Erlang C must match it.
* ``DepotConfig`` - the realistic depot: a finite fleet on shift schedules,
  distance-driven energy use, a site power cap and a selectable policy.
  It breaks the M/M/c assumptions on purpose, so Erlang C does not apply.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar


@dataclass(frozen=True)
class ValidationConfig:
    """M/M/c run. Time units are arbitrary; mean service time is 1 / mu."""

    lam: float = 4.0
    mu: float = 1.0
    c: int = 5
    horizon: float = 200_000.0
    warmup: float = 20_000.0

    mode: ClassVar[str] = "validation"

    def __post_init__(self) -> None:
        if not 0 <= self.warmup < self.horizon:
            raise ValueError("need 0 <= warmup < horizon")
        if self.lam / (self.c * self.mu) >= 1:
            raise ValueError("rho >= 1: the queue has no steady state")

    @property
    def rho(self) -> float:
        return self.lam / (self.c * self.mu)

    @classmethod
    def from_rho(cls, rho: float, c: int, mu: float = 1.0, **kwargs) -> "ValidationConfig":
        return cls(lam=rho * c * mu, mu=mu, c=c, **kwargs)

    def with_(self, **changes) -> "ValidationConfig":
        return replace(self, **changes)


@dataclass(frozen=True)
class DepotConfig:
    """Realistic depot run. Time unit is hours; day d starts at t = 24 d."""

    # Fleet
    n_vans: int = 50
    battery_kwh: float = 75.0
    kwh_per_km: float = 0.25
    distance_km: tuple[float, float] = (80.0, 200.0)  # uniform daily route length
    initial_soc: float = 1.0
    reserve_soc: float = 0.10  # buffer on top of the next route's energy

    # Shifts (hours)
    depart_window: tuple[float, float] = (6.0, 9.0)  # hour of day, uniform
    shift_hours: tuple[float, float] = (8.0, 10.0)  # time away, uniform

    # Depot
    n_chargers: int = 5
    charger_kw: float = 50.0
    site_cap_kw: float | None = 400.0  # None = no cap
    policy: str = "fcfs"  # fcfs | lowest_soc | earliest_departure

    # Run length
    n_days: int = 15
    warmup_days: int = 1  # day 0 starts with a full fleet; drop it

    mode: ClassVar[str] = "depot"

    def __post_init__(self) -> None:
        if self.site_cap_kw is not None and self.site_cap_kw < self.charger_kw:
            raise ValueError("site cap is below one charger's power: nothing can charge")
        if not 0 <= self.warmup_days < self.n_days:
            raise ValueError("need 0 <= warmup_days < n_days")
        lo, hi = self.distance_km
        if hi * self.kwh_per_km > self.battery_kwh:
            raise ValueError("longest route needs more than a full battery")

    def with_(self, **changes) -> "DepotConfig":
        return replace(self, **changes)
