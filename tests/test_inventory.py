"""Tests de la logique de reapprovisionnement (src/inventory.py)."""
import numpy as np
import pandas as pd
import pytest

from src.inventory import (calibrate_factors, factor_vector, recommend_order,
                           simulate, summarize)


def _frame(values, families):
    """DataFrame (series x dates) avec index (store_nbr, family)."""
    n, T = values.shape
    idx = pd.MultiIndex.from_arrays([np.arange(n), families], names=["store_nbr", "family"])
    return pd.DataFrame(values, index=idx, columns=pd.date_range("2017-01-01", periods=T))


def test_perfect_forecast_causes_no_stockout():
    Y = np.full((3, 28), 10.0)
    res = simulate(Y, Y.copy(), np.ones(3), lead_time=2, review_period=1,
                   holding_cost=0.1, stockout_cost=1.0)
    assert res["lost"].sum() == pytest.approx(0.0)
    # en regime etabli (horizon de prevision complet), on commande exactement la demande du jour
    assert res["ordered"][:, 5:25].mean() == pytest.approx(10.0)


def test_zero_forecast_loses_all_demand():
    Y = np.full((2, 14), 5.0)
    res = simulate(Y, np.zeros_like(Y), np.ones(2), 2, 1, 0.1, 1.0)
    assert res["lost"].sum() == pytest.approx(Y.sum())
    assert res["end_inv"].sum() == pytest.approx(0.0)


def test_more_safety_stock_never_increases_stockouts():
    rng = np.random.default_rng(0)
    F = rng.uniform(5, 15, size=(20, 28))
    Y = F * rng.uniform(0.6, 1.6, size=F.shape)
    lost = []
    for factor in (1.0, 1.3, 1.8):
        res = simulate(Y, F, np.full(20, factor), 2, 1, 0.1, 1.0)
        lost.append(res["lost"].sum())
    assert lost[0] >= lost[1] >= lost[2]


def test_costs_follow_definition():
    rng = np.random.default_rng(1)
    F = rng.uniform(5, 15, size=(5, 20))
    Y = F * rng.uniform(0.7, 1.4, size=F.shape)
    res = simulate(Y, F, np.full(5, 1.2), 2, 1, 0.1, 1.0)
    s = summarize(Y, res, 0.1, 1.0, warmup=2)
    assert s["cout_total"] == pytest.approx(s["cout_stockage"] + s["cout_rupture"])
    assert s["cout_rupture"] == pytest.approx(1.0 * res["lost"][:, 2:].sum())
    assert 0 <= s["taux_service_%"] <= 100


@pytest.mark.parametrize("lead,review", [(0, 1), (10, 5)])
def test_invalid_horizon_is_rejected(lead, review):
    Y = np.ones((1, 10))
    with pytest.raises(ValueError):
        simulate(Y, Y, np.ones(1), lead, review, 0.1, 1.0)


def test_recommend_order_arithmetic():
    rec = recommend_order([10, 10, 10, 10], factor=1.2, on_hand=5, on_order=5,
                          lead_time=2, review_period=1)
    assert rec["demande_prevue"] == pytest.approx(30)       # 3 premiers jours seulement
    assert rec["niveau_cible"] == pytest.approx(36)
    assert rec["stock_securite"] == pytest.approx(6)
    assert rec["quantite_a_commander"] == pytest.approx(26)


def test_recommend_order_is_never_negative():
    rec = recommend_order([10, 10, 10], 1.2, on_hand=1000, on_order=0,
                          lead_time=2, review_period=1)
    assert rec["quantite_a_commander"] == 0.0


def test_calibrated_factor_matches_forecast_error():
    F = np.full((10, 28), 10.0)
    fam = ["A"] * 10
    f1 = calibrate_factors(_frame(F, fam), _frame(F, fam), window=3, service_level=0.8)
    f2 = calibrate_factors(_frame(2 * F, fam), _frame(F, fam), window=3, service_level=0.8)
    assert f1["A"] == pytest.approx(1.0)    # prevision parfaite -> facteur 1
    assert f2["A"] == pytest.approx(2.0)    # demande = 2 x prevision -> facteur 2


def test_factor_vector_follows_family_order():
    idx = pd.MultiIndex.from_arrays([[1, 2, 3], ["B", "A", "Z"]], names=["store_nbr", "family"])
    vec = factor_vector({"A": 1.5, "B": 2.0}, idx, default=1.0)
    assert list(vec) == [2.0, 1.5, 1.0]