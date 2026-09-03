-- Распределение числа раундов: перцентили, доля игроков с 0 раундов
-- и выбросы (выше Q3 + 3 * IQR).
WITH q AS (
    SELECT
        version,
        count(*)                                     AS players,
        avg((sum_gamerounds = 0)::INT)               AS share_zero_rounds,
        avg(sum_gamerounds)                          AS mean_rounds,
        quantile_cont(sum_gamerounds, 0.25)          AS p25,
        quantile_cont(sum_gamerounds, 0.50)          AS median,
        quantile_cont(sum_gamerounds, 0.75)          AS p75,
        quantile_cont(sum_gamerounds, 0.90)          AS p90,
        quantile_cont(sum_gamerounds, 0.99)          AS p99,
        max(sum_gamerounds)                          AS max_rounds
    FROM cookie_cats
    GROUP BY version
)
SELECT
    q.*,
    q.p75 + 3 * (q.p75 - q.p25) AS outlier_threshold,
    (
        SELECT count(*)
        FROM cookie_cats AS c
        WHERE c.version = q.version
          AND c.sum_gamerounds > q.p75 + 3 * (q.p75 - q.p25)
    ) AS outliers
FROM q
ORDER BY version;
