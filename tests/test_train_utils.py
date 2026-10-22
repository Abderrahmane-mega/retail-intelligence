"""Tests des fonctions d'evaluation et du split chronologique (src/train.py)."""
import numpy as np
import pandas as pd
import pytest

from src.train import HORIZON_DAYS, compute_metrics, temporal_split


def test_metrics_known_values():
    y = np.array([10.0, 10.0])
    m = compute_metrics(y, np.array([12.0, 8.0]))
    assert m["MAE"] == pytest.approx(2.0)
    assert m["RMSE"] == pytest.approx(2.0)
    assert m["WAPE"] == pytest.approx(0.2)    # 4 / 20
    assert m["Biais"] == pytest.approx(0.0)   # sur- et sous-estimation se compensent


def test_negative_predictions_are_clipped_to_zero():
    m = compute_metrics(np.array([10.0]), np.array([-5.0]))
    assert m["MAE"] == pytest.approx(10.0)
    assert m["Biais"] == pytest.approx(-1.0)  # on sous-estime tout


def test_wape_is_undefined_when_total_sales_are_zero():
    m = compute_metrics(np.zeros(3), np.ones(3))
    assert np.isnan(m["WAPE"])


def test_temporal_split_has_no_overlap_and_expected_sizes():
    dates = pd.date_range("2017-01-01", "2017-06-30")
    df = pd.DataFrame({"date": np.repeat(dates, 2), "x": 1.0})
    train, val, test = temporal_split(df)
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()
    assert test["date"].max() == dates.max()
    assert val["date"].nunique() == HORIZON_DAYS
    assert test["date"].nunique() == HORIZON_DAYS
    assert len(train) + len(val) + len(test) == len(df)