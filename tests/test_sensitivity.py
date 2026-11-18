"""Directional checks: inputs move outputs the way they should.

Each comparison uses common random numbers (the same seeds on both sides), so
the difference reflects the input change rather than sampling noise.
"""

import numpy as np
import pytest

from depot_sim.config import DepotConfig
from depot_sim.runner import run_replications

N_RUNS = 5


def mean_of(cfg, key):
    return float(np.mean([r[key] for r in run_replications(cfg, n_runs=N_RUNS)]))


def test_more_chargers_less_waiting_when_power_is_not_binding():
    # No site cap here: with the 400 kW cap only 8 x 50 kW chargers can draw
    # at once, so a 12th charger cannot help (see the next test).
    waits = [mean_of(DepotConfig(n_chargers=c, site_cap_kw=None), "mean_wait") for c in (5, 8, 12)]
    assert waits[0] > waits[1] > waits[2]


def test_chargers_beyond_the_power_cap_do_not_help():
    w8 = mean_of(DepotConfig(n_chargers=8), "mean_wait")
    w12 = mean_of(DepotConfig(n_chargers=12), "mean_wait")
    assert w12 == pytest.approx(w8, rel=0.02)


def test_longer_routes_more_energy_fewer_on_time():
    base = DepotConfig(n_chargers=2)  # tight enough that on-time is below 100%
    short = base.with_(distance_km=(80.0, 200.0))
    long = base.with_(distance_km=(120.0, 260.0))
    assert mean_of(long, "energy_per_day_kwh") > mean_of(short, "energy_per_day_kwh")
    assert mean_of(long, "on_time") < mean_of(short, "on_time")


def test_doubling_charger_power_roughly_halves_charge_time():
    slow = mean_of(DepotConfig(charger_kw=50.0, site_cap_kw=None), "mean_charge_duration")
    fast = mean_of(DepotConfig(charger_kw=100.0, site_cap_kw=None), "mean_charge_duration")
    assert fast / slow == pytest.approx(0.5, rel=0.1)
