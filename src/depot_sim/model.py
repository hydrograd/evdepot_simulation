"""The SimPy charging model.

``ChargerBank`` is the queue machinery shared by both modes: a SimPy resource
for the queue plus a pool of charger ids, so utilisation can be tracked per
charger. ``run_validation`` drives it with M/M/c arrivals and service.
"""

from __future__ import annotations

import numpy as np
import simpy

from .config import ValidationConfig
from .metrics import BusyTracker, summarise_validation


class ChargerBank:
    """c identical chargers behind one queue.

    The SimPy resource decides who is served next; the id pool records which
    physical charger each job holds, so busy time is attributed per charger.
    """

    def __init__(
        self,
        env: simpy.Environment,
        n: int,
        tracker: BusyTracker,
        priority: bool = False,
    ) -> None:
        self.env = env
        self.n = n
        self.resource = simpy.PriorityResource(env, n) if priority else simpy.Resource(env, n)
        self.tracker = tracker
        self._free = list(range(n - 1, -1, -1))
        self._since: dict[int, float] = {}

    def request(self, priority: float | None = None):
        if priority is None or not isinstance(self.resource, simpy.PriorityResource):
            return self.resource.request()
        return self.resource.request(priority=priority)

    def acquire(self) -> int:
        """Call once a request has been granted; returns the charger id."""
        cid = self._free.pop()
        self._since[cid] = self.env.now
        return cid

    def release(self, req, cid: int) -> None:
        self.tracker.add(cid, self._since.pop(cid), self.env.now)
        self._free.append(cid)
        self.resource.release(req)


# --------------------------------------------------------------------------
# Validation mode: M/M/c
# --------------------------------------------------------------------------


def run_validation(cfg: ValidationConfig, rng: np.random.Generator) -> dict:
    """One M/M/c replication. Waits are recorded for jobs arriving after warm-up.

    A job's wait is recorded when its service starts, so jobs still queued
    when the horizon ends are not counted. With a horizon of many thousands
    of service times this end effect is negligible.
    """
    env = simpy.Environment()
    tracker = BusyTracker(cfg.c, cfg.warmup, cfg.horizon)
    bank = ChargerBank(env, cfg.c, tracker)
    waits: list[float] = []

    def job(arrival: float):
        req = bank.request()
        yield req
        cid = bank.acquire()
        if arrival >= cfg.warmup:
            waits.append(env.now - arrival)
        yield env.timeout(rng.exponential(1.0 / cfg.mu))
        bank.release(req, cid)

    def arrivals():
        while True:
            yield env.timeout(rng.exponential(1.0 / cfg.lam))
            env.process(job(env.now))

    env.process(arrivals())
    env.run(until=cfg.horizon)
    return summarise_validation(waits, tracker)
