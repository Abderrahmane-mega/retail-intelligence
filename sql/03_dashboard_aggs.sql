-- Agregations pour le dashboard
CREATE OR REPLACE TABLE features.sales_by_family_week AS
SELECT family, date_trunc('week', date) AS week, SUM(sales) AS sales
FROM clean.sales GROUP BY 1, 2;

CREATE OR REPLACE TABLE features.sales_by_store_week AS
SELECT store_nbr, date_trunc('week', date) AS week, SUM(sales) AS sales
FROM clean.sales GROUP BY 1, 2;
