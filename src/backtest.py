"""Backtest de la politique de reapprovisionnement sur la periode de test.

Usage : python src/backtest.py [--lead-time 2 --review-period 1 --holding-cost 0.1 ...]
Prerequis : python src/train.py execute (produit val_ et test_predictions.parquet).

Unite monetaire : 1 unite = marge perdue pour une unite demandee mais non servie.
Les couts sont des HYPOTHESES (aucun prix/marge dans le dataset) : voir la sensibilite.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inventory import (calibrate_factors, critical_ratio, factor_vector,  # noqa: E402
                           load_predictions, series_cost, simulate, summarize, to_matrix)

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def main(a) -> None:
    REPORTS.mkdir(exist_ok=True)
    L, R, h, p = a.lead_time, a.review_period, a.holding_cost, a.stockout_cost
    w, cr = L + R, critical_ratio(p, h)
    print(f"Hypotheses : delai={L}j, revue tous les {R}j, cout stockage={h}/unite/jour, "
          f"cout rupture={p}/unite, ratio critique={cr:.3f}, marge naive={a.naive_margin:.0%}")

    val, test = load_predictions(PROC / "val_predictions.parquet"), \
        load_predictions(PROC / "test_predictions.parquet")

    Y = to_matrix(test, "sales")
    idx, cols = Y.index, Y.columns
    mat = lambda df, c, ref: to_matrix(df, c).reindex(index=ref.index, columns=ref.columns)  # noqa: E731
    Fl, Fn = (mat(test, "lgbm", Y).values, mat(test, "naive_mean3", Y).values)
    assert not (np.isnan(Fl).any() or np.isnan(Fn).any() or np.isnan(Y.values).any()), \
        "valeurs manquantes dans les matrices de test"

    Yv = to_matrix(val, "sales")
    Fvl, Fvn = mat(val, "lgbm", Yv), mat(val, "naive_mean3", Yv)

    def factors(Fv, level):
        return factor_vector(calibrate_factors(Yv, Fv, w, level), idx)
    def pick_level(Fv, grid=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
        """Niveau de service qui minimise le cout simule sur la VALIDATION (test non utilise)."""
        costs = {}
        for lvl in grid:
            fac = factor_vector(calibrate_factors(Yv, Fv, w, lvl), Yv.index)
            res = simulate(Yv.values, Fv.values, fac, L, R, h, p)
            costs[lvl] = summarize(Yv.values, res, h, p, warmup=L)["cout_total"]
        return min(costs, key=costs.get), costs

    lvl_n, _ = pick_level(Fvn)
    lvl_l, costs_l = pick_level(Fvl)
    print(f"Niveau de service retenu sur la validation : naive={lvl_n}, lightgbm={lvl_l}")
    print("Cout simule en validation (lightgbm) :", {k: round(v) for k, v in costs_l.items()})
    ones = np.ones(len(idx))
    policies = {
        "1. Naive (sans securite)": (Fn, ones),
        f"2. Naive + marge {a.naive_margin:.0%}": (Fn, ones * (1 + a.naive_margin)),
        "3. Naive + securite calibree": (Fn, factors(Fvn, lvl_n)),
        "4. LightGBM (sans securite)": (Fl, ones),
        "5. LightGBM + securite calibree": (Fl, factors(Fvl, lvl_l)),
    }
    Yt = Y.values
    results, table = {}, {}
    for name, (F, fac) in policies.items():
        res = simulate(Yt, F, fac, L, R, h, p)
        results[name] = res
        table[name] = summarize(Yt, res, h, p, warmup=L)
    table = pd.DataFrame(table).T
    print(f"\n=== Backtest sur {Yt.shape[1]} jours de test ({len(idx)} series), "
          f"{L} jours d'echauffement exclus ===")
    print(table.to_string(float_format=lambda v: f"{v:,.1f}"))

    base = table.loc[f"2. Naive + marge {a.naive_margin:.0%}", "cout_total"]
    best = table.loc["5. LightGBM + securite calibree", "cout_total"]
    base_cal = table.loc["3. Naive + securite calibree", "cout_total"]
    print(f"\nCout total, politique 5 vs 2 (regle empirique) : {100 * (1 - best / base):+.1f} %"
          f" de reduction")
    print(f"Cout total, politique 5 vs 3 (meme methode de securite, prevision naive) : "
          f"{100 * (1 - best / base_cal):+.1f} % de reduction")
    table.to_csv(REPORTS / "replenishment_backtest.csv")

    # Par famille
    fam_cost = {}
    for name, res in results.items():
        c = pd.Series(series_cost(res, h, p, warmup=L), index=idx)
        fam_cost[name] = c.groupby(level="family").sum()
    fam = pd.DataFrame(fam_cost)
    fam["gain_5_vs_2_%"] = 100 * (1 - fam["5. LightGBM + securite calibree"]
                                  / fam[f"2. Naive + marge {a.naive_margin:.0%}"])
    print("\n=== Cout total par famille : gain de la politique 5 vs 2 ===")
    print(fam[["gain_5_vs_2_%"]].sort_values("gain_5_vs_2_%").to_string(
        float_format=lambda v: f"{v:.1f}"))
    fam.to_csv(REPORTS / "replenishment_by_family.csv")

    # Sensibilite au niveau de service vise
    rows = []
    for lvl in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]:
        for label, Fv, F in (("naive", Fvn, Fn), ("lgbm", Fvl, Fl)):
            res = simulate(Yt, F, factors(Fv, lvl), L, R, h, p)
            s = summarize(Yt, res, h, p, warmup=L)
            rows.append({"niveau_vise": lvl, "prevision": label, **s})
    sens = pd.DataFrame(rows)
    print("\n=== Sensibilite au niveau de service vise ===")
    print(sens[["niveau_vise", "prevision", "taux_service_%", "stock_moyen_total",
                "cout_total"]].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    sens.to_csv(REPORTS / "replenishment_sensitivity.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead-time", type=int, default=2)
    ap.add_argument("--review-period", type=int, default=1)
    ap.add_argument("--holding-cost", type=float, default=0.1)
    ap.add_argument("--stockout-cost", type=float, default=1.0)
    ap.add_argument("--naive-margin", type=float, default=0.15)
    main(ap.parse_args())