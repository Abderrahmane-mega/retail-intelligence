"""API de prevision et de recommandation de commande.

Lancer depuis la racine du projet :
    uvicorn api.main:app --reload
Documentation interactive : http://127.0.0.1:8000/docs

Mode retrospectif : les donnees s'arretent a la fin du jeu Favorita (15 aout 2017).
La prevision d'une date t n'utilise que des donnees anterieures a t-14.
"""
import datetime as dt
import json
import os
import sys
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import List

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inventory import (calibrate_factors, load_predictions,  # noqa: E402
                           recommend_order, to_matrix)

DATA_DIR = Path(os.getenv("RETAIL_DATA_DIR", ROOT / "data" / "processed"))
MODELS_DIR = Path(os.getenv("RETAIL_MODELS_DIR", ROOT / "models"))
DB_PATH = DATA_DIR / "retail.duckdb"
MAX_RANGE_DAYS = 60


# ----------------------------------------------------------------------------
# Chargement des ressources au demarrage
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    meta = json.loads((MODELS_DIR / "features.json").read_text())
    con = duckdb.connect(str(DB_PATH), read_only=True)
    val = load_predictions(DATA_DIR / "val_predictions.parquet")
    Yv = to_matrix(val, "sales")
    Fv = to_matrix(val, "lgbm").reindex(index=Yv.index, columns=Yv.columns)

    families = sorted(r[0] for r in con.execute(
        "SELECT DISTINCT family FROM features.sales_features").fetchall())
    stores = sorted(int(r[0]) for r in con.execute(
        "SELECT DISTINCT store_nbr FROM features.sales_features").fetchall())
    dmin, dmax = con.execute(
        "SELECT MIN(date), MAX(date) FROM features.sales_features").fetchone()

    app.state.booster = lgb.Booster(model_file=str(MODELS_DIR / "lgbm.txt"))
    app.state.features = meta["features"]
    app.state.cat_cols = meta["cat_cols"]
    app.state.con = con
    app.state.Yv, app.state.Fv = Yv, Fv
    app.state.families, app.state.stores = families, set(stores)
    app.state.date_range = (dmin, dmax)
    app.state.factor_cache = lru_cache(maxsize=64)(
        lambda window, level: calibrate_factors(Yv, Fv, window, level))
    yield
    con.close()


app = FastAPI(
    title="Retail Intelligence API",
    description="Prevision de la demande a 14 jours et recommandation de reapprovisionnement "
                "(mode retrospectif sur les donnees Favorita).",
    version="1.0.0", lifespan=lifespan)


# ----------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------
class ForecastPoint(BaseModel):
    date: dt.date
    prevision: float = Field(description="Ventes prevues (unites)")
    ventes_observees: float = Field(description="Ventes reelles (donnees historiques)")


class ForecastResponse(BaseModel):
    store_nbr: int
    family: str
    horizon_jours: int = 14
    points: List[ForecastPoint]


class RecommendRequest(BaseModel):
    store_nbr: int
    family: str
    date: dt.date = Field(description="Date de decision (la commande est passee ce jour-la)")
    on_hand: float = Field(0, ge=0, description="Stock disponible (unites)")
    on_order: float = Field(0, ge=0, description="Commandes en cours non livrees (unites)")
    lead_time: int = Field(2, ge=1, le=12, description="Delai de livraison (jours)")
    review_period: int = Field(1, ge=1, le=13, description="Periode de revue (jours)")
    service_level: float = Field(0.8, ge=0.5, le=0.99, description="Niveau de service vise")

    model_config = {"json_schema_extra": {"examples": [{
        "store_nbr": 44, "family": "DAIRY", "date": "2017-08-01", "on_hand": 120,
        "on_order": 0, "lead_time": 2, "review_period": 1, "service_level": 0.8}]}}

    @model_validator(mode="after")
    def check_horizon(self):
        if self.lead_time + self.review_period > 14:
            raise ValueError("lead_time + review_period doit etre <= 14 (horizon du modele)")
        return self


class RecommendResponse(BaseModel):
    store_nbr: int
    family: str
    date: dt.date
    horizon_jours: int
    jours_de_prevision_disponibles: int
    horizon_tronque: bool
    demande_prevue: float
    facteur_securite: float
    stock_securite: float
    niveau_cible: float
    quantite_a_commander: float


# ----------------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------------
def _check_series(state, store_nbr: int, family: str) -> None:
    if family not in state.families:
        raise HTTPException(404, f"Famille inconnue : {family}. Voir /metadata.")
    if store_nbr not in state.stores:
        raise HTTPException(404, f"Magasin inconnu : {store_nbr}. Voir /metadata.")


def _predict_series(state, store_nbr: int, family: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    cols = ", ".join(["date", "sales"] + state.features)
    df = state.con.cursor().execute(
        f"SELECT {cols} FROM features.sales_features "
        "WHERE store_nbr = ? AND family = ? AND date BETWEEN ? AND ? ORDER BY date",
        [store_nbr, family, start, end]).df()
    if df.empty:
        return df
    X = df[state.features].copy()
    for c in state.cat_cols:
        X[c] = X[c].astype("category")
    df["prevision"] = np.clip(np.expm1(state.booster.predict(X)), 0, None)
    return df


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/health")
def health(request: Request):
    s = request.app.state
    return {"status": "ok", "modele_charge": s.booster is not None,
            "donnees_du": str(s.date_range[0])[:10], "donnees_au": str(s.date_range[1])[:10]}


@app.get("/metadata")
def metadata(request: Request):
    s = request.app.state
    return {"familles": s.families, "magasins": sorted(s.stores),
            "date_min": str(s.date_range[0])[:10], "date_max": str(s.date_range[1])[:10]}


@app.get("/forecast", response_model=ForecastResponse)
def forecast(request: Request,
             store_nbr: int = Query(..., description="Numero de magasin"),
             family: str = Query(..., description="Famille de produits"),
             start: dt.date = Query(...), end: dt.date = Query(...)):
    s = request.app.state
    _check_series(s, store_nbr, family)
    if end < start:
        raise HTTPException(422, "end doit etre >= start")
    if (end - start).days + 1 > MAX_RANGE_DAYS:
        raise HTTPException(422, f"Plage limitee a {MAX_RANGE_DAYS} jours")
    df = _predict_series(s, store_nbr, family, start, end)
    if df.empty:
        raise HTTPException(404, f"Aucune donnee entre {start} et {end} "
                                 f"(donnees disponibles : {str(s.date_range[0])[:10]} "
                                 f"a {str(s.date_range[1])[:10]})")
    points = [ForecastPoint(date=r.date.date() if hasattr(r.date, "date") else r.date,
                            prevision=float(r.prevision), ventes_observees=float(r.sales))
              for r in df.itertuples()]
    return ForecastResponse(store_nbr=store_nbr, family=family, points=points)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, request: Request):
    s = request.app.state
    _check_series(s, req.store_nbr, req.family)
    w = req.lead_time + req.review_period
    df = _predict_series(s, req.store_nbr, req.family, req.date,
                         req.date + dt.timedelta(days=w - 1))
    if df.empty:
        raise HTTPException(404, f"Aucune prevision disponible a partir du {req.date} "
                                 f"(donnees jusqu'au {str(s.date_range[1])[:10]}).")
    factors = s.factor_cache(w, round(req.service_level, 2))
    factor = factors.get(req.family, 1.0)
    rec = recommend_order(df["prevision"].values, factor, req.on_hand, req.on_order,
                          req.lead_time, req.review_period)
    return RecommendResponse(
        store_nbr=req.store_nbr, family=req.family, date=req.date, horizon_jours=w,
        jours_de_prevision_disponibles=len(df), horizon_tronque=len(df) < w,
        demande_prevue=rec["demande_prevue"], facteur_securite=factor,
        stock_securite=rec["stock_securite"], niveau_cible=rec["niveau_cible"],
        quantite_a_commander=rec["quantite_a_commander"])