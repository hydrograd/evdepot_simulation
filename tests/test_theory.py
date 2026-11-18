import math

import pytest

from depot_sim import theory


@pytest.mark.parametrize("lam,mu", [(0.3, 1.0), (0.7, 1.0), (0.95, 1.0), (2.0, 5.0)])
def test_c1_reduces_to_mm1(lam, mu):
    rho = lam / mu
    assert theory.mean_wait_in_queue(lam, mu, 1) == pytest.approx(rho / (mu - lam), abs=1e-9)
    assert theory.erlang_c(lam, mu, 1) == pytest.approx(rho, abs=1e-12)


def test_wq_blows_up_as_rho_approaches_one():
    c, mu = 5, 1.0
    waits = [theory.mean_wait_in_queue(r * c * mu, mu, c) for r in (0.9, 0.99, 0.999, 0.9999)]
    assert all(b > 5 * a for a, b in zip(waits, waits[1:]))
    assert waits[-1] > 1_000


def test_wq_vanishes_at_light_load():
    assert theory.mean_wait_in_queue(0.01 * 5, 1.0, 5) < 1e-8


@pytest.mark.parametrize("rho", [1.0, 1.2])
def test_unstable_load_raises(rho):
    with pytest.raises(ValueError):
        theory.mean_wait_in_queue(rho * 3, 1.0, 3)


def test_closed_forms_for_small_c():
    # M/M/2 at rho = 0.5: P(wait) = 2 rho^2 / (1 + rho) = 1/3
    assert theory.erlang_c(1.0, 1.0, 2) == pytest.approx(1 / 3, abs=1e-12)
    # M/M/3 with a = 2: P0 = 1/9, P(wait) = 4/9
    assert theory.erlang_c(2.0, 1.0, 3) == pytest.approx(4 / 9, abs=1e-12)


def test_against_published_tables():
    # Erlang B table: 10 servers, 8 erlangs -> grade of service 0.1217
    assert theory.erlang_b(10, 8.0) == pytest.approx(0.1217, abs=5e-5)
    # Erlang C (call-centre tables): 10 agents, 8 erlangs -> P(wait) 0.4092
    assert theory.erlang_c(8.0, 1.0, 10) == pytest.approx(0.4092, abs=5e-5)


def test_recursion_is_stable_for_large_c():
    # The factorial form overflows a float well before c = 500.
    b = theory.erlang_b(500, 450.0)
    assert 0.0 < b < 1.0 and math.isfinite(b)


def test_littles_law_and_bundle_agree():
    r = theory.mmc(4.0, 1.0, 5)
    assert r.rho == pytest.approx(0.8)
    assert r.Lq == pytest.approx(r.lam * r.Wq)
    assert r.W == pytest.approx(r.Wq + 1.0)
    assert r.Wq == pytest.approx(theory.mean_wait_in_queue(4.0, 1.0, 5))
