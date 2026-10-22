"""Dashboard Retail Intelligence (Streamlit).

Lancer depuis la racine du projet :  streamlit run app/dashboard.py
Lit les fichiers produits par train.py et backtest.py (aucun reentrainement).
Mode retrospectif : les donnees s'arretent a la fin du jeu de test.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inventory import (calibrate_factors, factor_vector, load_predictions,  # noqa: E402
                           recommend_order, simulate, summarize, to_matrix)

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
DB_PATH = PROC / "retail.duckdb"

st.set_page_config(page_title="Retail Intelligence", layout="wide")

REQUIRED = [
    PROC / "test_predictions.parquet", PROC / "val_predictions.parquet",
    REPORTS / "metrics_overall.csv", REPORTS / "replenishment_backtest.csv",
    REPORTS / "replenishment_by_family.csv", REPORTS / "replenishment_sensitivity.csv",
]
missing = [str(f.relative_to(ROOT)) for f in REQUIRED if not f.exists()]
if missing:
    st.error("Fichiers manquants : " + ", ".join(missing)
             + ". Lancez d'abord `python src/train.py` puis `python src/backtest.py`.")
    st.stop()


# ----------------------------------------------------------------------------
# Chargement (mis en cache)
# ----------------------------------------------------------------------------
@st.cache_data
def load_pred(name: str) -> pd.DataFrame:
    return load_predictions(PROC / name)


@st.cache_data
def load_report(name: str, index_col=0) -> pd.DataFrame:
    return pd.read_csv(REPORTS / name, index_col=index_col)


@st.cache_data
def build_matrices():
    test, val = load_pred("test_predictions.parquet"), load_pred("val_predictions.parquet")

    def ref(df, col, base):
        return to_matrix(df, col).reindex(index=base.index, columns=base.columns)

    Y = to_matrix(test, "sales")
    Yv = to_matrix(val, "sales")
    return (Y, ref(test, "lgbm", Y), ref(test, "naive_mean3", Y),
            Yv, ref(val, "lgbm", Yv), ref(val, "naive_mean3", Yv))


@st.cache_data
def get_factors(which: str, window: int, level: float) -> dict:
    """Facteurs de securite par famille, calibres sur la VALIDATION."""
    _, _, _, Yv, Fvl, Fvn = build_matrices()
    return calibrate_factors(Yv, Fvl if which == "lgbm" else Fvn, window, level)


@st.cache_data
def load_history(family: str, store, start, end) -> pd.DataFrame:
    """Historique de ventes (optionnel) depuis DuckDB ; vide si indisponible."""
    try:
        import duckdb
        con = duckdb.connect(str(DB_PATH), read_only=True)
        q = ("SELECT date, SUM(sales) AS sales FROM clean.sales "
             "WHERE family = ? AND date >= ? AND date < ?")
        params = [family, start, end]
        if store != "Tous":
            q += " AND store_nbr = ?"
            params.append(int(store))
        df = con.execute(q + " GROUP BY date ORDER BY date", params).df()
        con.close()
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "sales"])


def policy_row(bt: pd.DataFrame, prefix: str) -> pd.Series:
    return bt[bt.index.str.startswith(prefix)].iloc[0]


# ----------------------------------------------------------------------------
# Page 1 : synthese
# ----------------------------------------------------------------------------
def page_summary():
    Y = build_matrices()[0]
    d0, d1 = Y.columns[0].date(), Y.columns[-1].date()
    st.title("Retail Intelligence")
    st.subheader("Prévision de la demande et optimisation des stocks")
    st.caption(f"Corporación Favorita (Kaggle) · {len(Y):,} séries magasin × famille · "
               f"prévision à 14 jours · backtest sur {Y.shape[1]} jours ({d0} → {d1})")

    mo = load_report("metrics_overall.csv")
    bt = load_report("replenishment_backtest.csv")
    p2, p3, p5 = policy_row(bt, "2."), policy_row(bt, "3."), policy_row(bt, "5.")

    c1, c2, c3, c4 = st.columns(4)
    w_l, w_n = mo.loc["lgbm", "WAPE"], mo.loc["naive_mean3", "WAPE"]
    c1.metric("Erreur de prévision (WAPE)", f"{w_l:.1%}",
              f"{w_l / w_n - 1:+.0%} vs baseline naïve", delta_color="inverse")
    c2.metric("Coût simulé", f"{p5['cout_total']:,.0f}",
              f"{p5['cout_total'] / p3['cout_total'] - 1:+.1%} vs naïve + sécurité",
              delta_color="inverse")
    c3.metric("Stock moyen", f"{p5['stock_moyen_total']:,.0f}",
              f"{p5['stock_moyen_total'] / p3['stock_moyen_total'] - 1:+.1%} vs naïve + sécurité",
              delta_color="inverse")
    c4.metric("Taux de service", f"{p5['taux_service_%']:.1f} %",
              f"{p5['taux_service_%'] - p3['taux_service_%']:+.1f} pt vs naïve + sécurité")
    st.info("Comparaison à procédure égale : même méthode de stock de sécurité, seule la "
            "prévision change. Les coûts sont des **hypothèses** (impact estimé par simulation, "
            "pas une économie réelle).")

    left, right = st.columns(2)
    with left:
        st.markdown("**Coût total simulé par politique**")
        fig = px.bar(bt.reset_index(names="politique"), x="cout_total", y="politique",
                     orientation="h", text_auto=".3s")
        fig.update_layout(yaxis_title=None, xaxis_title="Coût total (unités monétaires)",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig)
    with right:
        st.markdown("**Sensibilité au niveau de service visé**")
        sens = load_report("replenishment_sensitivity.csv", index_col=None)
        fig = px.line(sens, x="niveau_vise", y="cout_total", color="prevision", markers=True)
        fig.update_layout(xaxis_title="Niveau de service visé", yaxis_title="Coût total")
        st.plotly_chart(fig)

    st.markdown("**Résultats du backtest**")
    st.dataframe(bt.round(1))

    st.markdown("**Gain de coût par famille : LightGBM + sécurité vs règle empirique (%)**")
    fam = load_report("replenishment_by_family.csv")
    g = fam["gain_5_vs_2_%"].sort_values().reset_index()
    g.columns = ["famille", "gain"]
    g["signe"] = np.where(g["gain"] >= 0, "gain", "perte")
    fig = px.bar(g, x="gain", y="famille", orientation="h", color="signe",
                 color_discrete_map={"gain": "#2e8b57", "perte": "#c0392b"})
    fig.update_layout(height=700, yaxis_title=None, xaxis_title="Gain de coût (%)",
                      showlegend=False)
    st.plotly_chart(fig)

    with st.expander("Hypothèses et limites"):
        st.markdown(
            "- Aucun prix ni marge dans le jeu de données : coût de rupture = 1 par unité "
            "manquée, coût de stockage = 0,1 par unité et par jour (hypothèses réglables).\n"
            "- Ventes observées = demande (les ruptures passées ne sont pas corrigées).\n"
            "- Une seule fenêtre de test de 28 jours, hors période de fêtes.\n"
            "- Délai de livraison fixe ; pas de contrainte de capacité ni de péremption.\n"
            "- Le gain est concentré sur les familles à gros volume ; certaines petites "
            "familles sont moins bien servies par la politique calibrée.")


# ----------------------------------------------------------------------------
# Page 2 : previsions
# ----------------------------------------------------------------------------
def page_forecast():
    test = load_pred("test_predictions.parquet")
    st.title("Forecasting studio")
    with st.sidebar:
        st.header("Sélection")
        family = st.selectbox("Famille de produits", sorted(test["family"].unique()))
        store = st.selectbox("Magasin", ["Tous"] + sorted(test["store_nbr"].unique().tolist()))
        show_hist = st.checkbox("Afficher l'historique", value=True)

    d = test[test["family"] == family]
    if store != "Tous":
        d = d[d["store_nbr"] == store]
    agg = d.groupby("date")[["sales", "lgbm", "naive_mean3", "naive_lag14"]].sum().reset_index()

    fig = go.Figure()
    if show_hist:
        hist = load_history(family, store, (agg["date"].min() - pd.Timedelta(days=90)).date(),
                            agg["date"].min().date())
        if len(hist):
            fig.add_scatter(x=hist["date"], y=hist["sales"], name="Historique",
                            line=dict(color="#9aa0a6"))
    fig.add_scatter(x=agg["date"], y=agg["sales"], name="Ventes réelles",
                    line=dict(color="#1f2937", width=3))
    fig.add_scatter(x=agg["date"], y=agg["lgbm"], name="LightGBM", line=dict(color="#2563eb"))
    fig.add_scatter(x=agg["date"], y=agg["naive_mean3"], name="Baseline naïve (moy. 3 lags)",
                    line=dict(color="#f59e0b", dash="dot"))
    fig.update_layout(yaxis_title="Unités vendues", xaxis_title=None,
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig)

    st.markdown("**Erreurs sur la période de test** (calculées au niveau magasin × jour)")
    rows = {}
    for label, col in [("LightGBM", "lgbm"), ("Naïve (moy. 3 lags)", "naive_mean3"),
                       ("Naïve (lag 14)", "naive_lag14")]:
        err = d[col].clip(lower=0) - d["sales"]
        tot = d["sales"].sum()
        rows[label] = {"MAE": err.abs().mean(),
                       "RMSE": float(np.sqrt((err ** 2).mean())),
                       "WAPE": err.abs().sum() / tot if tot > 0 else np.nan,
                       "Biais": err.sum() / tot if tot > 0 else np.nan}
    res = pd.DataFrame(rows).T
    st.dataframe(res.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}",
                                   "WAPE": "{:.1%}", "Biais": "{:+.1%}"})
                 if hasattr(res, "style") else res)
    if len(d) and d["sales"].mean() < 5:
        st.caption("Volume moyen très faible (moins de 5 unités par magasin et par jour) : "
                   "le WAPE est peu fiable sur ce périmètre.")
    with st.expander("Données journalières"):
        st.dataframe(agg.round({"sales": 1, "lgbm": 1, "naive_mean3": 1, "naive_lag14": 1}))

    st.markdown("**Métriques globales (test)**")
    st.dataframe(load_report("metrics_overall.csv").round(4))


# ----------------------------------------------------------------------------
# Page 3 : simulateur
# ----------------------------------------------------------------------------
def page_simulator():
    Y, Fl, Fn, *_ = build_matrices()
    idx = Y.index
    fam_all = sorted(set(idx.get_level_values("family")))
    store_all = sorted(set(int(s) for s in idx.get_level_values("store_nbr")))

    st.title("Simulateur de réapprovisionnement")
    st.caption("Politique order-up-to : on commande pour ramener le stock au niveau cible "
               "= facteur de sécurité (calibré sur la validation) × prévisions sur "
               "(délai + période de revue) jours.")

    with st.sidebar:
        st.header("Périmètre")
        fam = st.selectbox("Famille", ["Toutes"] + fam_all)
        store = st.selectbox("Magasin", ["Tous"] + store_all)
        st.header("Logistique")
        L = st.slider("Délai de livraison (jours)", 1, 12, 2)
        R = st.slider("Période de revue (jours)", 1, 14 - L, 1)
        st.header("Coûts (hypothèses)")
        h = st.number_input("Stockage (par unité et par jour)", 0.0, 10.0, 0.1, 0.01)
        p = st.number_input("Rupture (par unité manquée)", 0.0, 100.0, 1.0, 0.1)
        st.header("Politique")
        level = st.slider("Niveau de service visé", 0.50, 0.99, 0.80, 0.01,
                          help="0,80 = niveau retenu par minimisation du coût sur la validation.")
        margin = st.slider("Marge de la règle empirique (%)", 0, 50, 15)
        shock = st.slider("Choc de demande (%)", -30, 30, 0,
                          help="Teste la politique si la demande réelle s'écarte des prévisions.")

    w = L + R
    mask = np.ones(len(idx), dtype=bool)
    if fam != "Toutes":
        mask &= np.asarray(idx.get_level_values("family") == fam)
    if store != "Tous":
        mask &= np.asarray(idx.get_level_values("store_nbr") == store)
    n = int(mask.sum())
    Yt = Y.values[mask] * (1 + shock / 100)
    if n == 0 or Yt.sum() == 0:
        st.warning("Aucune vente sur ce périmètre.")
        return

    fac_l = factor_vector(get_factors("lgbm", w, level), idx)[mask]
    fac_n = factor_vector(get_factors("naive", w, level), idx)[mask]
    ref_name = f"Règle empirique : naïve + marge {margin} %"
    policies = {
        ref_name: (Fn.values[mask], np.full(n, 1 + margin / 100)),
        "Naïve + sécurité calibrée": (Fn.values[mask], fac_n),
        "LightGBM + sécurité calibrée": (Fl.values[mask], fac_l),
    }
    out, sims = {}, {}
    for name, (F, fac) in policies.items():
        sims[name] = simulate(Yt, F, fac, L, R, h, p)
        out[name] = summarize(Yt, sims[name], h, p, warmup=L)
    tbl = pd.DataFrame(out).T
    ref = tbl.loc[ref_name]

    st.markdown(f"**{n} série(s)** · délai {L} j · revue tous les {R} j · "
                f"{Y.shape[1]} jours simulés ({L} jours d'échauffement exclus)")
    cols = st.columns(3)
    for col, name in zip(cols, policies):
        r = tbl.loc[name]
        col.markdown(f"**{name}**")
        is_ref = name == ref_name
        col.metric("Coût total", f"{r['cout_total']:,.0f}",
                   None if is_ref else f"{r['cout_total'] / ref['cout_total'] - 1:+.1%}",
                   delta_color="inverse")
        col.metric("Taux de service", f"{r['taux_service_%']:.1f} %",
                   None if is_ref else f"{r['taux_service_%'] - ref['taux_service_%']:+.1f} pt")
        col.metric("Stock moyen", f"{r['stock_moyen_total']:,.0f}",
                   None if is_ref else f"{r['stock_moyen_total'] / ref['stock_moyen_total'] - 1:+.1%}",
                   delta_color="inverse")
    st.caption("Variations affichées par rapport à la règle empirique.")

    dates = list(Y.columns)
    inv = pd.DataFrame({k: v["end_inv"].sum(axis=0) for k, v in sims.items()}, index=dates)
    lost = pd.DataFrame({k: v["lost"].sum(axis=0) for k, v in sims.items()}, index=dates)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Stock total en fin de journée**")
        st.plotly_chart(px.line(inv).update_layout(xaxis_title=None, yaxis_title="Unités",
                                                   legend_title=None,
                                                   legend=dict(orientation="h", y=-0.2)))
    with c2:
        st.markdown("**Unités manquées par jour**")
        st.plotly_chart(px.line(lost).update_layout(xaxis_title=None, yaxis_title="Unités",
                                                    legend_title=None,
                                                    legend=dict(orientation="h", y=-0.2)))
    with st.expander("Détail des indicateurs"):
        st.dataframe(tbl.round(1))

    st.divider()
    st.subheader("Recommandation de commande pour une série")
    st.caption("Mode rétrospectif : on se place à une date du test et on utilise les prévisions "
               "disponibles à ce moment-là (le modèle prévoit à 14 jours).")
    a, b, c = st.columns(3)
    rfam = a.selectbox("Famille", fam_all, key="rfam")
    rstore = b.selectbox("Magasin", store_all, key="rstore")
    rdate = c.selectbox("Date de décision", dates, format_func=lambda x: x.strftime("%d/%m/%Y"))
    e, f_ = st.columns(2)
    on_hand = e.number_input("Stock disponible (unités)", 0.0, value=0.0, step=10.0)
    on_order = f_.number_input("Commandes en cours (unités)", 0.0, value=0.0, step=10.0)

    key = (int(rstore), rfam)
    fc = Fl.loc[key].loc[rdate:].iloc[:w].values
    if len(fc) < w:
        st.warning(f"Seulement {len(fc)} jour(s) de prévision restant sur {w} nécessaires : "
                   "choisissez une date plus tôt dans la période.")
    rec = recommend_order(fc, get_factors("lgbm", w, level).get(rfam, 1.0),
                          on_hand, on_order, L, R)
    m = st.columns(4)
    m[0].metric(f"Demande prévue ({w} j)", f"{rec['demande_prevue']:,.1f}")
    m[1].metric("Stock de sécurité", f"{rec['stock_securite']:,.1f}")
    m[2].metric("Niveau cible", f"{rec['niveau_cible']:,.1f}")
    m[3].metric("Quantité à commander", f"{rec['quantite_a_commander']:,.1f}")
    actual = Y.loc[key].loc[rdate:].iloc[:w].sum()
    st.caption(f"Pour comparaison, demande réellement observée sur les mêmes {w} jours : "
               f"{actual:,.1f} (visible seulement parce que les données sont historiques).")


PAGES = {"Synthèse": page_summary, "Prévisions": page_forecast,
         "Simulateur de stock": page_simulator}
choice = st.sidebar.radio("Navigation", list(PAGES))
PAGES[choice]()