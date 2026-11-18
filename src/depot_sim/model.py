"""The SimPy charging model.

``ChargerBank`` is the queue machinery shared by both modes: a SimPy resource
for the queue plus a pool of charger ids, so utilisation can be tracked per
charger. ``run_validation`` drives it with M/M/c arrivals and service.
"""

from __future__ import annotations

import numpy as np
import simpy

from .config import DepotConfig, ValidationConfig
from .metrics import BusyTracker, summarise_depot, summarise_validation
from .policies import get_policy


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


# --------------------------------------------------------------------------
# Depot mode: finite fleet, shifts, power cap, policy
# --------------------------------------------------------------------------


class PowerMeter:
    """Site power drawn over time. Logged on every change, so it integrates exactly."""

    def __init__(self, env: simpy.Environment) -> None:
        self.env = env
        self.kw = 0.0
        self.peak = 0.0
        self.log: list[tuple[float, float]] = [(0.0, 0.0)]

    def change(self, delta_kw: float) -> None:
        self.kw += delta_kw
        self.peak = max(self.peak, self.kw)
        self.log.append((self.env.now, self.kw))

    def energy_kwh(self) -> float:
        t = np.array([p[0] for p in self.log])
        kw = np.array([p[1] for p in self.log])
        return float(np.sum(kw[:-1] * np.diff(t)))


def _schedule(cfg: DepotConfig, rng: np.random.Generator):
    """Departure and return times (hours) and route lengths, drawn up front.

    Shape (n_vans, n_days + 1): day n_days is the morning after the last
    simulated day; its departure is the final charging deadline and its route
    sets the final readiness requirement.
    """
    shape = (cfg.n_vans, cfg.n_days + 1)
    days = np.arange(cfg.n_days + 1) * 24.0
    depart = days + rng.uniform(*cfg.depart_window, size=shape)
    back = depart + rng.uniform(*cfg.shift_hours, size=shape)
    dist = rng.uniform(*cfg.distance_km, size=shape)
    return depart, back, dist


def run_depot(cfg: DepotConfig, rng: np.random.Generator, keep_logs: bool = False) -> dict:
    """One replication of the depot over ``cfg.n_days`` days.

    Each evening a van returns, queues for a charger (in policy order), then
    for site power, then charges at constant power until full or until its
    next departure, whichever comes first. A finished van is unplugged and the
    charger freed at once. Charging is linear (no taper above ~80% SoC).
    """
    env = simpy.Environment()
    policy = get_policy(cfg.policy)
    window = (24.0 * cfg.warmup_days, 24.0 * cfg.n_days)
    tracker = BusyTracker(cfg.n_chargers, *window)
    bank = ChargerBank(env, cfg.n_chargers, tracker, priority=policy.uses_priority)
    power = (
        simpy.Container(env, init=cfg.site_cap_kw, capacity=cfg.site_cap_kw)
        if cfg.site_cap_kw is not None
        else None
    )
    meter = PowerMeter(env)
    depart, back, dist = _schedule(cfg, rng)
    cap = cfg.battery_kwh
    kw = cfg.charger_kw

    sessions: list[dict] = []
    departures: list[dict] = []
    trips: list[dict] = []

    def required_soc(van: int, day: int) -> float:
        return min(1.0, dist[van, day] * cfg.kwh_per_km / cap + cfg.reserve_soc)

    def charge(van: int, day: int, soc: float):
        """Charge after day ``day``'s shift; returns the SoC at the deadline."""
        arrival, deadline = env.now, depart[van, day + 1]
        rec = {
            "van": van, "day": day, "arrival": arrival, "deadline": deadline,
            "soc_in": soc, "charger": None, "got_charger": False, "got_power": False,
            "wait_charger": deadline - arrival, "wait": deadline - arrival,
            "start": None, "duration": 0.0, "energy_kwh": 0.0,
        }
        req = bank.request(policy.priority(soc, deadline))
        yield req | env.timeout(max(0.0, deadline - env.now))
        if not req.triggered:  # deadline passed while queueing for a charger
            req.cancel()
            rec["soc_out"] = soc
            sessions.append(rec)
            return soc
        cid = bank.acquire()
        rec.update(charger=cid, got_charger=True, wait_charger=env.now - arrival,
                   hold_start=env.now)

        if power is not None:
            get = power.get(kw)
            yield get | env.timeout(max(0.0, deadline - env.now))
            if not get.triggered:  # deadline passed while waiting for site power
                get.cancel()
                bank.release(req, cid)
                rec.update(soc_out=soc, hold_end=env.now)
                sessions.append(rec)
                return soc
        rec.update(got_power=True, start=env.now, wait=env.now - arrival)

        duration = min((1.0 - soc) * cap / kw, max(0.0, deadline - env.now))
        meter.change(+kw)
        yield env.timeout(duration)
        meter.change(-kw)
        energy = kw * duration
        soc = soc + energy / cap  # not clipped: test_conservation checks it stays <= 1
        if power is not None:
            yield power.put(kw)
        bank.release(req, cid)
        rec.update(duration=duration, energy_kwh=energy, soc_out=soc, hold_end=env.now)
        sessions.append(rec)
        return soc

    def van_life(van: int):
        soc = cfg.initial_soc
        for day in range(cfg.n_days + 1):
            yield env.timeout(max(0.0, depart[van, day] - env.now))
            need = required_soc(van, day)
            departures.append({"van": van, "day": day, "soc": soc, "required": need,
                               "ready": soc >= need - 1e-9})
            if day == cfg.n_days:
                return
            energy_need = dist[van, day] * cfg.kwh_per_km
            used = min(energy_need, soc * cap)
            trips.append({"van": van, "day": day, "need_kwh": energy_need,
                          "used_kwh": used, "stranded": used < energy_need - 1e-9})
            soc -= used / cap
            yield env.timeout(back[van, day] - env.now)
            soc = yield env.process(charge(van, day, soc))

    for v in range(cfg.n_vans):
        env.process(van_life(v))
    env.run()  # every van process ends after the final departure

    out = summarise_depot(cfg, sessions, departures, trips, tracker, meter.peak)
    if keep_logs:
        out["_logs"] = {
            "sessions": sessions, "departures": departures, "trips": trips,
            "power_log": meter.log, "metered_energy_kwh": meter.energy_kwh(),
            "final_soc": [d["soc"] for d in departures if d["day"] == cfg.n_days],
        }
    return out
