-- Минимальный n на группу из сетки, при котором мощность >= 80%.
WITH power AS (
    SELECT uplift, n_per_group, avg(reject::INT) AS power
    FROM sims
    WHERE scenario = 'power'
    GROUP BY ALL
)
SELECT
    uplift,
    MIN(n_per_group) FILTER (WHERE power >= 0.8) AS min_n_power_80,
    max(power)                                   AS max_power_in_grid
FROM power
GROUP BY uplift
ORDER BY uplift;
