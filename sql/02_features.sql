-- Couche FEATURES. Horizon H = 14 jours, modele direct :
-- toutes les variables derivees des ventes sont decalees d au moins 14 jours
-- => aucune fuite de donnees, une seule prevision pour les 14 prochains jours.
CREATE SCHEMA IF NOT EXISTS features;

CREATE OR REPLACE TABLE features.sales_features AS
WITH tx AS (
    SELECT date, store_nbr,
        LAG(transactions, 14) OVER q AS tx_lag_14,
        AVG(transactions) OVER (q ROWS BETWEEN 20 PRECEDING AND 14 PRECEDING) AS tx_mean_7,
        AVG(transactions) OVER (q ROWS BETWEEN 40 PRECEDING AND 14 PRECEDING) AS tx_mean_27
    FROM clean.transactions
    WINDOW q AS (PARTITION BY store_nbr ORDER BY date)
),
oil AS (
    SELECT date, LAG(oil_price, 14) OVER (ORDER BY date) AS oil_lag_14
    FROM clean.oil
),
w AS (
    SELECT
        s.date, s.store_nbr, s.family, s.sales, s.onpromotion,
        st.type    AS store_type,
        st.cluster AS store_cluster,
        COALESCE(h.is_holiday, 0) AS is_holiday,
        dayofweek(s.date)  AS dow,
        day(s.date)        AS dom,
        month(s.date)      AS month,
        LAG(s.sales, 14) OVER p AS lag_14,
        LAG(s.sales, 21) OVER p AS lag_21,
        LAG(s.sales, 28) OVER p AS lag_28,
        AVG(s.sales) OVER (p ROWS BETWEEN 20 PRECEDING AND 14 PRECEDING) AS roll_mean_7,
        AVG(s.sales) OVER (p ROWS BETWEEN 40 PRECEDING AND 14 PRECEDING) AS roll_mean_27,
        STDDEV(s.sales) OVER (p ROWS BETWEEN 40 PRECEDING AND 14 PRECEDING) AS roll_std_27,
        AVG(s.onpromotion) OVER (p ROWS BETWEEN 20 PRECEDING AND 14 PRECEDING) AS promo_mean_7,
        tx.tx_lag_14, tx.tx_mean_7, tx.tx_mean_27,
        oil.oil_lag_14
    FROM clean.sales s
    LEFT JOIN raw.stores st USING (store_nbr)
    LEFT JOIN clean.national_holidays h ON h.date = s.date
    LEFT JOIN tx  ON tx.date = s.date AND tx.store_nbr = s.store_nbr
    LEFT JOIN oil ON oil.date = s.date
    WINDOW p AS (PARTITION BY s.store_nbr, s.family ORDER BY s.date)
)
SELECT * FROM w WHERE lag_28 IS NOT NULL AND roll_mean_27 IS NOT NULL;
