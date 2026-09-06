-- Рейтинг тестов по мощности в каждой точке, где эффект есть.
WITH power AS (
    SELECT scenario, metric, test, n_per_group, uplift,
           regexp_replace(param, ';bootstrap$', '') AS setting,
           avg(reject::INT) AS power
    FROM sims
    WHERE uplift > 0 AND scenario IN ('heavy_tails', 'ratio', 'cuped')
    GROUP BY ALL
)
SELECT
    *,
    RANK() OVER (PARTITION BY scenario, setting, n_per_group ORDER BY power DESC) AS power_rank
FROM power
ORDER BY scenario, setting, power_rank;
