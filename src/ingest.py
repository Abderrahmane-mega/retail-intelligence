"""Ingestion reproductible : CSV -> DuckDB (raw -> clean -> features).

Usage:  python -m src.ingest
Prerequis : telecharger les CSV de Kaggle "Store Sales - Time Series Forecasting"
dans data/raw/ (train.csv, stores.csv, oil.csv, holidays_events.csv, transactions.csv).
"""
from pathlib import Path
import logging
import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "processed" / "retail.duckdb"
SQL_DIR = ROOT / "sql"

TABLES = {
    "train": "train.csv",
    "stores": "stores.csv",
    "oil": "oil.csv",
    "holidays_events": "holidays_events.csv",
    "transactions": "transactions.csv",
}
EXPECTED_COLS = {
    "train": {"date", "store_nbr", "family", "sales", "onpromotion"},
    "stores": {"store_nbr", "city", "state", "type", "cluster"},
    "transactions": {"date", "store_nbr", "transactions"},
    "oil": {"date", "dcoilwtico"},
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")


def load_raw(con):
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    for table, fname in TABLES.items():
        path = RAW / fname
        if not path.exists():
            raise FileNotFoundError(f"Fichier manquant : {path}")
        con.execute(
            f"CREATE OR REPLACE TABLE raw.{table} AS "
            f"SELECT * FROM read_csv_auto('{path.as_posix()}')"
        )
        n = con.execute(f"SELECT COUNT(*) FROM raw.{table}").fetchone()[0]
        log.info("raw.%s chargee : %s lignes", table, f"{n:,}")


def validate(con):
    for table, expected in EXPECTED_COLS.items():
        cols = {r[0] for r in con.execute(f"DESCRIBE raw.{table}").fetchall()}
        missing = expected - cols
        if missing:
            raise ValueError(f"raw.{table} : colonnes manquantes {missing}")
    dups = con.execute(
        "SELECT COUNT(*) FROM (SELECT date, store_nbr, family FROM raw.train "
        "GROUP BY 1,2,3 HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    nulls = con.execute(
        "SELECT COUNT(*) FROM raw.train WHERE sales IS NULL OR date IS NULL"
    ).fetchone()[0]
    neg = con.execute("SELECT COUNT(*) FROM raw.train WHERE sales < 0").fetchone()[0]
    log.info("Validation : doublons=%s, nulls=%s, ventes negatives=%s", dups, nulls, neg)
    if nulls or neg:
        raise ValueError("Donnees invalides : nulls ou ventes negatives dans raw.train")


def run_sql_files(con):
    for f in sorted(SQL_DIR.glob("*.sql")):
        log.info("Execution de %s", f.name)
        con.execute(f.read_text(encoding="utf-8"))


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    load_raw(con)
    validate(con)
    run_sql_files(con)
    n = con.execute("SELECT COUNT(*) FROM features.sales_features").fetchone()[0]
    log.info("features.sales_features : %s lignes. Termine.", f"{n:,}")
    con.close()


if __name__ == "__main__":
    main()
