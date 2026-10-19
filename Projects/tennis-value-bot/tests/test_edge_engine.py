import pytest

from src.edge_engine import evaluate, kelly_stake, taker_fee_per_share, vwap_fill_price

CFG = {
    "bankroll_usd": 200, "min_edge": 0.03, "buffer": 0.01, "kelly_fraction": 0.25,
    "min_stake": 2, "max_stake": 10, "max_total_exposure": 100,
    "max_daily_new_exposure": 50, "max_positions_per_tournament_per_day": 5,
    "min_volume_usd": 5000, "min_minutes_to_start": 10, "price_band": [0.10, 0.90],
    "depth_multiple": 3, "gas_cost_usd": 0.02, "taker_fee_rate": 0.03,
}


def _eval(fair=0.60, asks=None, depth=1000.0, volume=10000.0, mins=120.0,
          has_pos=False, daily=0.0, open_exp=0.0, tpos=0):
    return evaluate("a", "Player", fair, asks or [(0.50, 10000)], depth, volume,
                    mins, has_pos, daily, open_exp, tpos, CFG)


def test_taker_fee_peak_at_half():
    assert taker_fee_per_share(0.5, CFG) == pytest.approx(0.0075)
    assert taker_fee_per_share(0.1, CFG) == pytest.approx(0.0027)


def test_vwap_walks_the_book():
    asks = [(0.50, 10), (0.55, 100)]  # $5 at 0.50 then deep at 0.55
    price, filled = vwap_fill_price(asks, 10.0)
    assert filled == pytest.approx(10.0)
    assert 0.50 < price < 0.55


def test_kelly_positive_edge_sizes_up():
    s = kelly_stake(0.60, 0.50, CFG)
    assert s > 0
    assert s <= CFG["max_stake"]


def test_kelly_negative_edge_zero():
    assert kelly_stake(0.45, 0.50, CFG) == 0.0


def test_bet_fires_on_clean_edge():
    sig = _eval()  # fair .60 vs ask .50 -> huge edge
    assert sig.action == "bet"
    assert sig.net_edge >= CFG["min_edge"]
    assert sig.limit_price <= sig.fair - CFG["min_edge"]


def test_skip_small_edge():
    sig = _eval(fair=0.52)  # gross 0.02 < costs+min_edge
    assert sig.action == "skip" and "net edge" in sig.reason


def test_skip_price_band():
    sig = _eval(asks=[(0.95, 10000)])
    assert sig.action == "skip" and "band" in sig.reason


def test_skip_low_volume():
    sig = _eval(volume=100)
    assert sig.action == "skip" and "volume" in sig.reason


def test_skip_near_start():
    sig = _eval(mins=5)
    assert sig.action == "skip" and "min to start" in sig.reason


def test_skip_existing_position():
    sig = _eval(has_pos=True)
    assert sig.action == "skip" and "existing position" in sig.reason


def test_skip_thin_depth():
    sig = _eval(depth=1.0)
    assert sig.action == "skip" and "depth" in sig.reason


def test_skip_exposure_caps():
    assert _eval(daily=49.0).action == "skip"
    assert _eval(open_exp=99.0).action == "skip"
    assert _eval(tpos=5).action == "skip"


def test_fee_included_in_net_edge():
    sig = _eval(fair=0.60, asks=[(0.55, 10000)])
    fee = taker_fee_per_share(0.55, CFG)
    expected_net = (0.60 - 0.55) - CFG["buffer"] - fee - CFG["gas_cost_usd"] / sig.stake
    assert sig.net_edge == pytest.approx(expected_net)
