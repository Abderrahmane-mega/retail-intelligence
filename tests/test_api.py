"""Tests de l'API. Necessitent les artefacts produits par ingest.py et train.py
(ils sont ignores automatiquement s'ils sont absents)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [ROOT / "models" / "lgbm.txt", ROOT / "models" / "features.json",
            ROOT / "data" / "processed" / "retail.duckdb",
            ROOT / "data" / "processed" / "val_predictions.parquet",
            ROOT / "data" / "processed" / "test_predictions.parquet"]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in REQUIRED),
                                reason="artefacts manquants : lancer ingest.py et train.py")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample():
    test = pd.read_parquet(ROOT / "data" / "processed" / "test_predictions.parquet")
    row = test.iloc[len(test) // 2]
    return test, int(row.store_nbr), str(row.family)


def _body(sample, **over):
    test, store, fam = sample
    body = {"store_nbr": store, "family": fam, "date": str(pd.Timestamp(test.date.min()).date()),
            "on_hand": 50, "on_order": 0, "lead_time": 2, "review_period": 1,
            "service_level": 0.8}
    body.update(over)
    return body


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_forecast_matches_predictions_saved_by_training(client, sample):
    test, store, fam = sample
    d0 = pd.Timestamp(test.date.min())
    d1 = d0 + pd.Timedelta(days=6)
    r = client.get("/forecast", params={"store_nbr": store, "family": fam,
                                        "start": str(d0.date()), "end": str(d1.date())})
    assert r.status_code == 200
    api = pd.DataFrame(r.json()["points"])
    api["date"] = pd.to_datetime(api["date"])
    ref = test[(test.store_nbr == store) & (test.family == fam)
               & test.date.between(d0, d1)][["date", "lgbm"]]
    merged = api.merge(ref, on="date")
    assert len(merged) == len(ref) > 0
    assert np.allclose(merged["prevision"], merged["lgbm"].clip(lower=0), atol=1e-6)


def test_recommend_is_internally_consistent(client, sample):
    r = client.post("/recommend", json=_body(sample))
    assert r.status_code == 200
    j = r.json()
    assert j["quantite_a_commander"] >= 0
    assert j["niveau_cible"] == pytest.approx(j["facteur_securite"] * j["demande_prevue"])
    assert j["stock_securite"] == pytest.approx(j["niveau_cible"] - j["demande_prevue"])


def test_more_stock_on_hand_means_smaller_order(client, sample):
    q0 = client.post("/recommend", json=_body(sample, on_hand=0)).json()["quantite_a_commander"]
    q1 = client.post("/recommend", json=_body(sample, on_hand=100)).json()["quantite_a_commander"]
    assert q0 >= q1 >= 0


def test_unknown_family_returns_404(client, sample):
    assert client.post("/recommend", json=_body(sample, family="XXX")).status_code == 404


@pytest.mark.parametrize("over", [{"lead_time": 12, "review_period": 5}, {"on_hand": -1},
                                  {"service_level": 1.5}])
def test_invalid_inputs_return_422(client, sample, over):
    assert client.post("/recommend", json=_body(sample, **over)).status_code == 422


def test_date_outside_data_returns_404(client, sample):
    assert client.post("/recommend", json=_body(sample, date="2030-01-01")).status_code == 404


def test_horizon_is_flagged_as_truncated_at_the_end_of_data(client, sample):
    last = client.get("/metadata").json()["date_max"]
    j = client.post("/recommend", json=_body(sample, date=last)).json()
    assert j["horizon_tronque"] is True