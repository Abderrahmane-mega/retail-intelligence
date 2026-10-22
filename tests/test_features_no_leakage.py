"""Tests du pipeline SQL sur un mini jeu de donnees synthetique :
- les variables sont bien definies (lags, moyennes glissantes) ;
- AUCUNE fuite temporelle : une valeur modifiee au jour t ne change aucune variable
  avant le jour t + 14 ;
- le calendrier est complete (jours manquants comptes comme zero).
"""
import datetime as dt
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SQL_FILES = [ROOT / "sql" / "01_clean.sql", ROOT / "sql" / "02_features.sql"]
N_DAYS = 90
START = dt.date(2017, 1, 1)
STORES, FAMILIES = (1, 2), ("A", "B")


def make_raw(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    dates = [START + dt.timedelta(d) for d in range(N_DAYS)]
    train = pd.DataFrame(
        [(d, s, f, float(rng.integers(0, 50)), int(rng.integers(0, 4)))
         for d in dates for s in STORES for f in FAMILIES],
        columns=["date", "store_nbr", "family", "sales", "onpromotion"])
    return {
        "train": train,
        "stores": pd.DataFrame({"store_nbr": [1, 2], "city": ["Q", "Q"], "state": ["P", "P"],
                                "type": ["A", "B"], "cluster": [1, 2]}),
        "holidays_events": pd.DataFrame({
            "date": [START + dt.timedelta(10)], "type": ["Holiday"], "locale": ["National"],
            "locale_name": ["E"], "description": ["x"], "transferred": ["False"]}),
        "transactions": pd.DataFrame(
            [(d, s, int(rng.integers(300, 900))) for d in dates for s in STORES],
            columns=["date", "store_nbr", "transactions"]),
        "oil": pd.DataFrame({
            "date": dates,
            "dcoilwtico": pd.array([50.0 + i * 0.1 if i % 7 not in (5, 6) else None
                                    for i in range(N_DAYS)], dtype="Float64")}),
    }


def build_db(raw: dict) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA raw")
    for name, df in raw.items():
        con.register("tmp", df)
        con.execute(f"CREATE TABLE raw.{name} AS SELECT * FROM tmp")
        con.unregister("tmp")
    for f in SQL_FILES:
        con.execute(f.read_text(encoding="utf-8"))
    return con


def features(raw: dict) -> pd.DataFrame:
    con = build_db(raw)
    out = con.execute("SELECT * FROM features.sales_features "
                      "ORDER BY store_nbr, family, date").df()
    con.close()
    out["date"] = pd.to_datetime(out["date"])
    return out


def test_lags_and_rolling_means_match_definition():
    raw = make_raw()
    f = features(raw)
    s = raw["train"].query("store_nbr == 1 and family == 'A'").copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.set_index("date")["sales"]
    r = f[(f.store_nbr == 1) & (f.family == "A")].set_index("date")
    assert np.allclose(r["lag_14"], s.shift(14).reindex(r.index))
    assert np.allclose(r["lag_28"], s.shift(28).reindex(r.index))
    assert np.allclose(r["roll_mean_7"], s.rolling(7).mean().shift(14).reindex(r.index))
    assert np.allclose(r["roll_mean_27"],
                       s.rolling(27, min_periods=1).mean().shift(14).reindex(r.index))


@pytest.mark.parametrize("table,col", [("train", "sales"), ("transactions", "transactions")])
def test_future_values_do_not_leak_into_features(table, col):
    raw = make_raw()
    base = features(raw)
    t0 = START + dt.timedelta(60)

    raw2 = {k: v.copy() for k, v in raw.items()}
    df = raw2[table]
    mask = (df["date"] == t0) & (df["store_nbr"] == 1)
    if table == "train":
        mask &= df["family"] == "A"
    df.loc[mask, col] = df.loc[mask, col] + 1000
    changed = features(raw2)

    cutoff = pd.Timestamp(t0 + dt.timedelta(14))
    cols = [c for c in base.columns if c not in ("date", "store_nbr", "family", "sales")]
    before_a, before_b = base[base.date < cutoff], changed[changed.date < cutoff]
    pd.testing.assert_frame_equal(before_a[cols].reset_index(drop=True),
                                  before_b[cols].reset_index(drop=True))
    # controle positif : a partir de t + 14 la modification est bien visible
    after_a, after_b = base[base.date >= cutoff], changed[changed.date >= cutoff]
    assert not after_a[cols].reset_index(drop=True).equals(after_b[cols].reset_index(drop=True))


def test_missing_calendar_day_is_completed_with_zero():
    raw = make_raw()
    missing = START + dt.timedelta(30)
    raw["train"] = raw["train"][raw["train"]["date"] != missing]
    raw["transactions"] = raw["transactions"][raw["transactions"]["date"] != missing]
    con = build_db(raw)
    n = con.execute("SELECT COUNT(*) FROM clean.sales").fetchone()[0]
    filled = con.execute("SELECT SUM(sales), COUNT(*) FROM clean.sales WHERE date = ?",
                         [missing]).fetchone()
    tx = con.execute("SELECT SUM(transactions) FROM clean.transactions WHERE date = ?",
                     [missing]).fetchone()[0]
    con.close()
    assert n == N_DAYS * len(STORES) * len(FAMILIES)
    assert filled == (0, len(STORES) * len(FAMILIES))
    assert tx == 0


def test_features_have_no_missing_values_in_core_columns():
    f = features(make_raw())
    core = ["lag_14", "lag_21", "lag_28", "roll_mean_7", "roll_mean_27",
            "tx_lag_14", "tx_mean_7", "tx_mean_27", "oil_lag_14"]
    assert not f[core].isna().any().any()