import pytest

from src.fair_value import devig_two_way, quote_ok

CFG = {"max_overround": 1.08, "max_quote_age_sec": 7200, "max_prob_jump": 0.08}


def test_devig_known_values():
    # 1.90/1.90 -> 50/50 regardless of vig
    fv = devig_two_way(1.90, 1.90)
    assert fv.fair_a == pytest.approx(0.5)
    assert fv.fair_b == pytest.approx(0.5)
    assert fv.overround == pytest.approx(2 / 1.90)


def test_devig_asymmetric():
    fv = devig_two_way(1.25, 4.20)
    assert fv.fair_a + fv.fair_b == pytest.approx(1.0)
    assert fv.fair_a == pytest.approx((1 / 1.25) / (1 / 1.25 + 1 / 4.20))
    assert fv.fair_a > 0.75


def test_devig_no_vig_passthrough():
    fv = devig_two_way(2.0, 2.0)
    assert fv.fair_a == pytest.approx(0.5)
    assert fv.overround == pytest.approx(1.0)


def test_devig_rejects_bad_odds():
    with pytest.raises(ValueError):
        devig_two_way(0.95, 2.0)


def test_gate_overround():
    fv = devig_two_way(1.80, 1.80)  # overround ~1.111
    ok, why = quote_ok(fv, 10, None, CFG)
    assert not ok and "overround" in why


def test_gate_stale_quote():
    fv = devig_two_way(1.95, 1.95)
    ok, why = quote_ok(fv, 9000, None, CFG)
    assert not ok and "age" in why


def test_gate_prob_jump():
    fv = devig_two_way(1.95, 1.95)  # fair_a = 0.5
    ok, why = quote_ok(fv, 10, 0.40, CFG)
    assert not ok and "jumped" in why


def test_gate_passes_clean_quote():
    fv = devig_two_way(1.95, 1.95)
    ok, _ = quote_ok(fv, 10, 0.49, CFG)
    assert ok
