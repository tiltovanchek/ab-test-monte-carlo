-- Гистограмма p-value в A/A-тестах (20 корзин по 0.05). При верной H0 она ровная.
WITH binned AS (
    SELECT metric, test, n_per_group, least(floor(p_value * 20), 19) / 20.0 AS bin_start
    FROM sims
    WHERE scenario = 'aa'
),
counts AS (
    SELECT metric, test, n_per_group, bin_start, count(*) AS cnt
    FROM binned
    GROUP BY metric, test, n_per_group, bin_start
)
SELECT
    *,
    cnt / sum(cnt) OVER (PARTITION BY metric, test, n_per_group) AS share
FROM counts
ORDER BY metric, test, n_per_group, bin_start;
