-- Во сколько раз CUPED уменьшает дисперсию оценки эффекта. Теория: 1 - rho^2.
-- Интервал приближённый: SE логарифма отношения дисперсий около sqrt(4 / R).
WITH v AS (
    SELECT
        CAST(regexp_extract(param, 'rho=([\d.]+)', 1) AS DOUBLE) AS rho,
        uplift,
        count(*) FILTER (WHERE test = 'welch_t')        AS reps,
        var_samp(effect) FILTER (WHERE test = 'cuped')   AS var_cuped,
        var_samp(effect) FILTER (WHERE test = 'welch_t') AS var_plain,
        avg(reject::INT) FILTER (WHERE test = 'cuped')   AS rate_cuped,
        avg(reject::INT) FILTER (WHERE test = 'welch_t') AS rate_plain
    FROM sims
    WHERE scenario = 'cuped'
    GROUP BY ALL
)
SELECT
    rho, uplift, reps,
    var_cuped / var_plain                                   AS var_ratio,
    var_cuped / var_plain * exp(-1.959964 * sqrt(4.0 / reps)) AS var_ratio_low,
    var_cuped / var_plain * exp(1.959964 * sqrt(4.0 / reps))  AS var_ratio_high,
    1 - rho * rho                                           AS var_ratio_theory,
    rate_plain,
    rate_cuped
FROM v
ORDER BY uplift, rho;
