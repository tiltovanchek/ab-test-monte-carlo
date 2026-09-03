-- Retention 1-го и 7-го дня по группам: доля, стандартная ошибка и 95% интервал.
WITH long AS (
    SELECT version, 'retention_1' AS metric, retention_1::INT AS retained FROM cookie_cats
    UNION ALL
    SELECT version, 'retention_7', retention_7::INT FROM cookie_cats
),
agg AS (
    SELECT metric, version, count(*) AS n, avg(retained) AS rate
    FROM long
    GROUP BY metric, version
)
SELECT
    metric,
    version,
    n,
    round(rate, 5)                                  AS rate,
    round(sqrt(rate * (1 - rate) / n), 5)           AS se,
    round(rate - 1.959964 * sqrt(rate * (1 - rate) / n), 5) AS ci_low,
    round(rate + 1.959964 * sqrt(rate * (1 - rate) / n), 5) AS ci_high
FROM agg
ORDER BY metric, version;
