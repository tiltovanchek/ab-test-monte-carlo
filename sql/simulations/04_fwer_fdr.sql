-- Множественное тестирование. На эксперимент и поправку две строки:
-- metric = 'null' (метрики без эффекта) и 'true' (с эффектом), effect - сколько из них значимы.
-- FWER - доля экспериментов с хоть одной ложной находкой,
-- FDR - средняя доля ложных среди найденных, power - доля найденных настоящих эффектов.
WITH per_rep AS (
    SELECT
        param, test, rep,
        bool_or(reject) FILTER (WHERE metric = 'null')   AS any_false,
        coalesce(sum(effect) FILTER (WHERE metric = 'null'), 0) AS false_found,
        coalesce(sum(effect) FILTER (WHERE metric = 'true'), 0) AS true_found
    FROM sims
    WHERE scenario = 'multiple'
    GROUP BY ALL
),
fdp AS (
    SELECT *, false_found / greatest(false_found + true_found, 1) AS fdp
    FROM per_rep
)
SELECT
    param,
    CAST(regexp_extract(param, 'k=(\d+)', 1) AS INT)         AS k,
    CAST(regexp_extract(param, 'rho=([\d.]+)', 1) AS DOUBLE) AS rho,
    CAST(regexp_extract(param, 'true=(\d+)', 1) AS INT)      AS n_true,
    test,
    count(*)                                   AS reps,
    avg(any_false::INT)                        AS fwer,
    wilson_lo(sum(any_false::INT), count(*))   AS fwer_low,
    wilson_hi(sum(any_false::INT), count(*))   AS fwer_high,
    avg(fdp)                                   AS fdr,
    avg(fdp) - 1.959964 * stddev_samp(fdp) / sqrt(count(*)) AS fdr_low,
    avg(fdp) + 1.959964 * stddev_samp(fdp) / sqrt(count(*)) AS fdr_high,
    avg(true_found) / nullif(max(CAST(regexp_extract(param, 'true=(\d+)', 1) AS INT)), 0) AS power
FROM fdp
GROUP BY ALL
ORDER BY n_true, rho, k, test;
