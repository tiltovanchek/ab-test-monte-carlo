-- A/A-тест с выпадением малоактивных пользователей из B: как часто t-тест
-- "находит" рост и как часто хи-квадрат ловит перекос (порог 0.001 и 0.05).
SELECT
    CAST(regexp_extract(param, 'drop=([\d.]+)', 1) AS DOUBLE) AS drop_rate,
    test,
    count(*)                                      AS reps,
    avg(reject::INT)                              AS rate,
    wilson_lo(sum(reject::INT), count(*))         AS ci_low,
    wilson_hi(sum(reject::INT), count(*))         AS ci_high,
    avg((p_value < 0.05)::INT)                    AS rate_at_005,
    avg(effect)                                   AS mean_effect
FROM sims
WHERE scenario = 'srm' AND metric = 'activity'
GROUP BY ALL
ORDER BY drop_rate, test;
