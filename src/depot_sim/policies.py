"""Charging policies: who gets the next free charger.

FCFS uses a plain ``simpy.Resource``. The other two use
``simpy.PriorityResource``, where a LOWER priority value is served first and
ties fall back to request order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Policy:
    name: str
    description: str
    key: Callable[[float, float], float] | None  # (soc_on_arrival, next_departure) -> priority

    @property
    def uses_priority(self) -> bool:
        return self.key is not None

    def priority(self, soc_on_arrival: float, next_departure: float) -> float | None:
        return None if self.key is None else self.key(soc_on_arrival, next_departure)


POLICIES: dict[str, Policy] = {
    "fcfs": Policy("fcfs", "first come, first served", None),
    "lowest_soc": Policy("lowest_soc", "lowest state of charge first", lambda soc, dep: soc),
    "earliest_departure": Policy(
        "earliest_departure", "earliest next departure first", lambda soc, dep: dep
    ),
}


def get_policy(name: str) -> Policy:
    try:
        return POLICIES[name]
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}") from None
