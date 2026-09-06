-- Покрытие: доля интервалов, в которые попала истинная разница. Для 95% интервала около 0.95.
SELECT
    scenario, metric, test, n_per_group, uplift,
    regexp_replace(param, ';bootstrap$', '') AS setting,
    count(*)                                  AS reps,
    avg(ci_covers::INT)                       AS coverage,
    wilson_lo(sum(ci_covers::INT), count(*))  AS ci_low,
    wilson_hi(sum(ci_covers::INT), count(*))  AS ci_high
FROM sims
WHERE ci_covers IS NOT NULL
GROUP BY ALL
ORDER BY scenario, metric, setting, test, n_per_group, uplift;
