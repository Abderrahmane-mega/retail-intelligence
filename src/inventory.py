"""Logique de reapprovisionnement : stock de securite calibre + simulation.

Politique "order-up-to" (base-stock) avec revue periodique :
  - tous les R jours, on commande pour ramener la position de stock
    (stock physique + commandes en cours) au niveau cible S ;
  - la commande arrive L jours plus tard ;
  - S = facteur_famille x somme des previsions sur L+R jours ;
  - facteur_famille = quantile (niveau de service) du ratio ventes reelles /
    ventes prevues sur la validation (donc jamais calibre sur le test).

Contrainte : L + R <= 14, car le modele prevoit a 14 jours : la prevision du jour
t n'utilise que des donnees anterieures a t-14, donc connue au moment de decider.
"""
import numpy as np
import pandas as pd


def critical_ratio(stockout_cost: float, holding_cost: float) -> float:
    """Ratio critique du newsvendor : Cu / (Cu + Co)."""
    return stockout_cost / (stockout_cost + holding_cost)


# ----------------------------------------------------------------------------
# Chargement et mise en forme
# ----------------------------------------------------------------------------
def load_predictions(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["store_nbr"] = df["store_nbr"].astype(int)
    df["family"] = df["family"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns:
        if c in ("lgbm", "naive_mean3", "naive_lag14", "ridge"):
            df[c] = df[c].clip(lower=0)
    return df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)


def to_matrix(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Format long -> matrice (series x dates)."""
    return df.pivot_table(index=["store_nbr", "family"], columns="date",
                          values=col, aggfunc="first")


# ----------------------------------------------------------------------------
# Calibration du stock de securite
# ----------------------------------------------------------------------------
def _rolling_sum(a: np.ndarray, w: int) -> np.ndarray:
    cs = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, axis=1)], axis=1)
    return cs[:, w:] - cs[:, :-w]


def calibrate_factors(Yv: pd.DataFrame, Fv: pd.DataFrame, window: int,
                      service_level: float, min_forecast: float = 1.0) -> dict:
    """Facteur multiplicatif par famille = quantile du ratio reel/prevu agrege sur
    `window` jours, calcule sur la validation."""
    Y, F = Yv.values, Fv.reindex_like(Yv).values
    ry, rf = _rolling_sum(Y, window), _rolling_sum(F, window)
    ratio = np.where(rf >= min_forecast, ry / np.where(rf > 0, rf, 1.0), np.nan)
    fam = np.array(Yv.index.get_level_values("family"))
    global_q = float(np.nanquantile(ratio, service_level))
    out = {}
    for f in np.unique(fam):
        r = ratio[fam == f].ravel()
        r = r[~np.isnan(r)]
        q = float(np.quantile(r, service_level)) if r.size >= 30 else global_q
        out[f] = float(np.clip(q, 0.5, 10.0))
    return out


def factor_vector(factors: dict, index: pd.MultiIndex, default: float = 1.0) -> np.ndarray:
    fam = index.get_level_values("family")
    return np.array([factors.get(f, default) for f in fam], dtype=float)


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def simulate(Y: np.ndarray, F: np.ndarray, factor: np.ndarray, lead_time: int,
             review_period: int, holding_cost: float, stockout_cost: float) -> dict:
    """Simule le stock de chaque serie jour par jour (ventes perdues si rupture).
    Y : demande reelle, F : prevision (series x jours), factor : (series,)."""
    if lead_time < 1:
        raise ValueError("lead_time doit etre >= 1")
    if lead_time + review_period > 14:
        raise ValueError("lead_time + review_period doit etre <= 14 (horizon du modele)")
    n, T = Y.shape
    w = lead_time + review_period
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(F, axis=1)], axis=1)
    d_idx = np.arange(T)
    win = cs[:, np.minimum(d_idx + w, T)] - cs[:, d_idx]   # previsions sur [d, d+w[
    S = factor[:, None] * win                              # niveau cible

    on_hand = S[:, 0].copy()
    on_order = np.zeros(n)
    arrivals = np.zeros((n, T + lead_time + 1))
    end_inv, lost, ordered = (np.zeros((n, T)) for _ in range(3))

    for d in range(T):
        arr = arrivals[:, d]
        on_hand += arr
        on_order -= arr
        if d % review_period == 0:
            q = np.maximum(S[:, d] - (on_hand + on_order), 0.0)
            arrivals[:, d + lead_time] += q
            on_order += q
            ordered[:, d] = q
        sold = np.minimum(on_hand, Y[:, d])
        lost[:, d] = Y[:, d] - sold
        on_hand -= sold
        end_inv[:, d] = on_hand
    return {"end_inv": end_inv, "lost": lost, "ordered": ordered}


def series_cost(res: dict, holding_cost: float, stockout_cost: float, warmup: int = 0):
    sl = slice(warmup, None)
    return (holding_cost * res["end_inv"][:, sl].sum(axis=1)
            + stockout_cost * res["lost"][:, sl].sum(axis=1))


def summarize(Y: np.ndarray, res: dict, holding_cost: float, stockout_cost: float,
              warmup: int = 0) -> dict:
    sl = slice(warmup, None)
    demand = Y[:, sl].sum()
    lost = res["lost"][:, sl]
    inv = res["end_inv"][:, sl]
    hold = holding_cost * inv.sum()
    short = stockout_cost * lost.sum()
    return {
        "taux_service_%": 100 * (1 - lost.sum() / demand),
        "taux_rupture_%": 100 * (lost > 1e-6).mean(),
        "stock_moyen_total": inv.sum(axis=0).mean(),
        "unites_manquees": lost.sum(),
        "cout_stockage": hold,
        "cout_rupture": short,
        "cout_total": hold + short,
    }


# ----------------------------------------------------------------------------
# Recommandation pour une serie (utilisee par l'API / le dashboard)
# ----------------------------------------------------------------------------
def recommend_order(forecast_next_days, factor: float, on_hand: float, on_order: float,
                    lead_time: int, review_period: int) -> dict:
    w = lead_time + review_period
    base = float(np.sum(np.asarray(forecast_next_days, dtype=float)[:w]))
    target = factor * base
    qty = max(0.0, target - (on_hand + on_order))
    return {"demande_prevue": base, "stock_securite": target - base,
            "niveau_cible": target, "quantite_a_commander": qty}