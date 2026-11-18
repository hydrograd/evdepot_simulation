"""Depot-mode invariants.

Each check compares two quantities recorded by independent code paths, so a
bug in one shows up as a mismatch:

* site power metered over time (PowerMeter) vs energy booked per session vs
  the change in each van's state of charge;
* a fleet-wide energy balance: start + charged - driven = end;
* the site power cap, with a control showing the cap actually binds;
* charger occupancy rebuilt from session records (no charger double-booked).
"""

import numpy as np
import pytest

from depot_sim.config import DepotConfig
from depot_sim.model import run_depot

CONFIGS = {
    "default": DepotConfig(),
    "cap_binding": DepotConfig(n_chargers=12),  # 12 x 50 kW > 400 kW cap
    "tight": DepotConfig(n_chargers=2, distance_km=(150.0, 250.0)),  # vans leave short
    "lowest_soc": DepotConfig(policy="lowest_soc", n_chargers=3),
    "earliest_departure": DepotConfig(policy="earliest_departure", n_chargers=3),
}


@pytest.fixture(scope="module", params=list(CONFIGS))
def run(request):
    cfg = CONFIGS[request.param]
    return cfg, run_depot(cfg, np.random.default_rng(7), keep_logs=True)


def test_metered_energy_matches_sessions_and_battery_gain(run):
    cfg, out = run
    logs = out["_logs"]
    booked = sum(s["energy_kwh"] for s in logs["sessions"])
    battery_gain = sum((s["soc_out"] - s["soc_in"]) * cfg.battery_kwh for s in logs["sessions"])
    assert booked > 0
    assert logs["metered_energy_kwh"] == pytest.approx(booked, rel=1e-9)
    assert battery_gain == pytest.approx(booked, rel=1e-9)


def test_fleet_energy_balance(run):
    cfg, out = run
    logs = out["_logs"]
    start = cfg.n_vans * cfg.initial_soc * cfg.battery_kwh
    charged = sum(s["energy_kwh"] for s in logs["sessions"])
    driven = sum(t["used_kwh"] for t in logs["trips"])
    end = sum(logs["final_soc"]) * cfg.battery_kwh
    assert start + charged - driven == pytest.approx(end, rel=1e-9)


def test_site_power_never_exceeds_cap(run):
    cfg, out = run
    drawn = np.array([kw for _, kw in out["_logs"]["power_log"]])
    assert drawn.max() <= cfg.site_cap_kw + 1e-9
    assert out["peak_power_kw"] <= cfg.site_cap_kw + 1e-9
    assert drawn.min() >= -1e-9


def test_cap_is_not_vacuous():
    # With 12 chargers the cap binds: peak sits exactly on it ...
    capped = run_depot(DepotConfig(n_chargers=12), np.random.default_rng(7))
    assert capped["peak_power_kw"] == pytest.approx(400.0)
    # ... and without the cap the same fleet draws more than 400 kW.
    free = run_depot(DepotConfig(n_chargers=12, site_cap_kw=None), np.random.default_rng(7))
    assert free["peak_power_kw"] > 400.0


def test_charger_utilisation_in_unit_interval(run):
    cfg, out = run
    util = np.array(out["charger_utilisation"])
    assert util.shape == (cfg.n_chargers,)
    assert np.all((util >= 0) & (util <= 1))
    assert util.sum() > 0


def test_no_charger_is_double_booked(run):
    cfg, out = run
    held = [s for s in out["_logs"]["sessions"] if s["got_charger"]]
    for cid in range(cfg.n_chargers):
        spans = sorted((s["hold_start"], s["hold_end"]) for s in held if s["charger"] == cid)
        for (_, end), (nxt, _) in zip(spans, spans[1:]):
            assert nxt >= end - 1e-9, f"charger {cid} double-booked"
    # Rebuilt occupancy never exceeds the number of chargers.
    events = sorted([(s["hold_start"], 1) for s in held] + [(s["hold_end"], -1) for s in held],
                    key=lambda e: (e[0], e[1]))
    assert max(np.cumsum([e[1] for e in events])) <= cfg.n_chargers


def test_state_of_charge_stays_in_bounds(run):
    cfg, out = run
    for s in out["_logs"]["sessions"]:
        assert 0.0 <= s["soc_in"] <= s["soc_out"] <= 1.0 + 1e-9
    for d in out["_logs"]["departures"]:
        assert 0.0 <= d["soc"] <= 1.0 + 1e-9


def test_tight_depot_actually_misses_departures():
    # Guards against the invariants above passing only because nothing is stressed.
    out = run_depot(CONFIGS["tight"], np.random.default_rng(7))
    assert out["on_time"] < 0.9
