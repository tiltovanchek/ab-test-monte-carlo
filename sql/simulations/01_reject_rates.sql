-- Доля отвержений H0 в каждой точке: при uplift = 0 это FPR, при uplift > 0 мощность.
-- Интервал Уилсона 95%.
SELECT
    scenario, metric, test, n_per_group, uplift, param,
    count(*)                                    AS reps,
    sum(reject::INT)                            AS rejects,
    avg(reject::INT)                            AS rate,
    wilson_lo(sum(reject::INT), count(*))       AS ci_low,
    wilson_hi(sum(reject::INT), count(*))       AS ci_high
FROM sims
WHERE scenario <> 'multiple'
GROUP BY ALL
ORDER BY scenario, metric, test, n_per_group, uplift, param;
