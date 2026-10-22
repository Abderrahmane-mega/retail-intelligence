"""Entrainement et evaluation : baselines, Ridge, LightGBM (horizon 14 jours).

Usage :
    python src/train.py            # complet (3M lignes)
    python src/train.py --quick    # test rapide sur les magasins 1 a 5

Prerequis : python src/ingest.py (ou python -m src.ingest) deja execute.
"""
import argparse
import json
import logging
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "retail.duckdb"
PRED_PATH = ROOT / "data" / "processed" / "test_predictions.parquet"
VAL_PRED_PATH = ROOT / "data" / "processed" / "val_predictions.parquet"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

# ----------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------
TARGET = "sales"
CAT_COLS = ["family", "store_nbr", "store_type", "store_cluster", "dow", "month"]
# Variables en "unites de ventes / volumes" : on leur applique log1p pour Ridge
NUM_LOG = [
    "lag_14", "lag_21", "lag_28",
    "roll_mean_7", "roll_mean_27", "roll_std_27",
    "onpromotion", "promo_mean_7",
    "tx_lag_14", "tx_mean_7", "tx_mean_27",
]
NUM_OTHER = ["is_holiday", "dom", "oil_lag_14"]
FEATURES = CAT_COLS + NUM_LOG + NUM_OTHER

HORIZON_DAYS = 28          # taille de la validation et du test (multiple de 7)
RIDGE_ALPHAS = [1, 100, 1000]
LGBM_PARAMS = dict(
    objective="regression",    # L2 sur la cible log1p ~ RMSLE
    learning_rate=0.05,
    n_estimators=2000,         # plafond ; l'arret anticipe decide
    num_leaves=63,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


# ----------------------------------------------------------------------------
# Donnees et split chronologique
# ----------------------------------------------------------------------------
def load_data(quick: bool) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    cols = ", ".join(["date", TARGET] + FEATURES)
    where = "WHERE store_nbr <= 5" if quick else ""
    df = con.execute(f"SELECT {cols} FROM features.sales_features {where}").df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    log.info("Donnees chargees : %s lignes, %s series", f"{len(df):,}",
             df.groupby(["store_nbr", "family"], observed=True).ngroups)
    return df


def temporal_split(df: pd.DataFrame):
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=HORIZON_DAYS - 1)
    val_start = test_start - pd.Timedelta(days=HORIZON_DAYS)
    train = df[df["date"] < val_start]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start]
    log.info("Train : %s -> %s (%s lignes)", train["date"].min().date(),
             train["date"].max().date(), f"{len(train):,}")
    log.info("Val   : %s -> %s (%s lignes)", val["date"].min().date(),
             val["date"].max().date(), f"{len(val):,}")
    log.info("Test  : %s -> %s (%s lignes)", test["date"].min().date(),
             test["date"].max().date(), f"{len(test):,}")
    return train, val, test


# ----------------------------------------------------------------------------
# Metriques
# ----------------------------------------------------------------------------
def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 0, None)
    err = p - y
    total = y.sum()
    return {
        "MAE": np.abs(err).mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "WAPE": np.abs(err).sum() / total if total > 0 else np.nan,
        "Biais": err.sum() / total if total > 0 else np.nan,
        "RMSLE": np.sqrt(((np.log1p(p) - np.log1p(y)) ** 2).mean()),
    }


def wape_by(df: pd.DataFrame, pred_cols: list, by: str) -> pd.DataFrame:
    out = {}
    for name in pred_cols:
        tmp = df.assign(ae=(df[name].clip(lower=0) - df[TARGET]).abs())
        g = tmp.groupby(by, observed=True)
        out[name] = g["ae"].sum() / g[TARGET].sum().replace(0, np.nan)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Modeles
# ----------------------------------------------------------------------------
def build_ridge(alpha: float) -> Pipeline:
    log_tf = make_pipeline(SimpleImputer(strategy="median"),
                           FunctionTransformer(np.log1p),
                           StandardScaler())
    num_tf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    pre = ColumnTransformer([
        ("log", log_tf, NUM_LOG),
        ("num", num_tf, NUM_OTHER),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))])


def fit_ridge(train, val):
    y_tr, y_va = np.log1p(train[TARGET].values), val[TARGET].values
    best = None
    for a in RIDGE_ALPHAS:
        model = build_ridge(a).fit(train[FEATURES], y_tr)
        pred = np.expm1(model.predict(val[FEATURES]))
        w = compute_metrics(y_va, pred)["WAPE"]
        log.info("Ridge alpha=%s -> WAPE validation = %.4f", a, w)
        if best is None or w < best[0]:
            best = (w, a, model)
    log.info("Ridge : meilleur alpha = %s", best[1])
    return best[2], best[1]


def fit_lgbm(train, val):
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        train[FEATURES], np.log1p(train[TARGET].values),
        eval_set=[(val[FEATURES], np.log1p(val[TARGET].values))],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    log.info("LightGBM : %s arbres retenus", model.best_iteration_)
    return model


# ----------------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------------
def main(quick: bool) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_data(quick)
    train, val, test = temporal_split(df)

    # Baselines (aucun apprentissage)
    test = test.copy()
    test["naive_lag14"] = test["lag_14"]
    test["naive_mean3"] = test[["lag_14", "lag_21", "lag_28"]].mean(axis=1)

    ridge, best_alpha = fit_ridge(train, val)
    test["ridge"] = np.expm1(ridge.predict(test[FEATURES]))

    lgbm = fit_lgbm(train, val)
    test["lgbm"] = np.expm1(lgbm.predict(test[FEATURES]))

    # Evaluation finale sur le test (touche une seule fois)
    models = ["naive_lag14", "naive_mean3", "ridge", "lgbm"]
    y = test[TARGET].values
    overall = pd.DataFrame({m: compute_metrics(y, test[m].values) for m in models}).T
    print("\n=== Resultats sur le TEST (28 derniers jours) ===")
    print(overall.to_string(float_format=lambda v: f"{v:.4f}"))

    by_family = wape_by(test, models, "family")
    by_family["gain_vs_naive_%"] = 100 * (1 - by_family["lgbm"] / by_family["naive_lag14"])
    print("\n=== WAPE par famille (test) ===")
    print(by_family.sort_values("lgbm").to_string(float_format=lambda v: f"{v:.3f}"))

    by_dow = wape_by(test.assign(dow=test["dow"].astype(int)), models, "dow")
    print("\n=== WAPE par jour de la semaine (test) ===")
    print(by_dow.to_string(float_format=lambda v: f"{v:.3f}"))

    imp = pd.Series(lgbm.booster_.feature_importance("gain"), index=FEATURES)
    imp = (100 * imp / imp.sum()).sort_values(ascending=False)
    print("\n=== Importance des variables LightGBM (% du gain) ===")
    print(imp.to_string(float_format=lambda v: f"{v:.1f}"))

    # Sauvegardes
    overall.to_csv(REPORTS_DIR / "metrics_overall.csv")
    by_family.to_csv(REPORTS_DIR / "metrics_by_family.csv")
    by_dow.to_csv(REPORTS_DIR / "metrics_by_dow.csv")
    imp.to_csv(REPORTS_DIR / "feature_importance.csv", header=["gain_pct"])
    joblib.dump(ridge, MODELS_DIR / "ridge.joblib")
    lgbm.booster_.save_model(str(MODELS_DIR / "lgbm.txt"))
    (MODELS_DIR / "features.json").write_text(json.dumps({
        "features": FEATURES, "cat_cols": CAT_COLS,
        "ridge_alpha": best_alpha, "best_iteration": int(lgbm.best_iteration_),
        "horizon_days": 14, "target_transform": "log1p",
    }, indent=2))
    keep = ["date", "store_nbr", "family", TARGET] + models
    test[keep].to_parquet(PRED_PATH, index=False)
    # Previsions de validation : servent a calibrer le stock de securite (backtest)
    val = val.copy()
    val["lgbm"] = np.expm1(lgbm.predict(val[FEATURES]))
    val["naive_mean3"] = val[["lag_14", "lag_21", "lag_28"]].mean(axis=1)
    val[["date", "store_nbr", "family", TARGET, "lgbm", "naive_mean3"]].to_parquet(
        VAL_PRED_PATH, index=False)
    log.info("Termine. Modeles dans %s, rapports dans %s", MODELS_DIR, REPORTS_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="test rapide (magasins 1 a 5)")
    main(ap.parse_args().quick)