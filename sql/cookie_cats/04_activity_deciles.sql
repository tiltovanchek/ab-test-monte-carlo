-- Retention 7-го дня по децилям активности: где разница между группами больше всего.
-- Число раундов измерено уже после начала теста, поэтому такое деление может давать смещение.
WITH deciles AS (
    SELECT
        *,
        NTILE(10) OVER (PARTITION BY version ORDER BY sum_gamerounds) AS decile
    FROM cookie_cats
),
by_group AS (
    SELECT
        decile,
        version,
        count(*)                  AS players,
        min(sum_gamerounds)       AS min_rounds,
        max(sum_gamerounds)       AS max_rounds,
        avg(retention_7::INT)     AS retention_7
    FROM deciles
    GROUP BY decile, version
),
wide AS (
    SELECT
        decile,
        max(min_rounds) FILTER (WHERE version = 'gate_30')  AS rounds_from,
        max(max_rounds) FILTER (WHERE version = 'gate_30')  AS rounds_to,
        max(retention_7) FILTER (WHERE version = 'gate_30') AS ret7_gate_30,
        max(retention_7) FILTER (WHERE version = 'gate_40') AS ret7_gate_40
    FROM by_group
    GROUP BY decile
)
SELECT
    decile,
    rounds_from,
    rounds_to,
    round(ret7_gate_30, 4)                AS ret7_gate_30,
    round(ret7_gate_40, 4)                AS ret7_gate_40,
    round(ret7_gate_40 - ret7_gate_30, 4) AS diff,
    RANK() OVER (ORDER BY abs(ret7_gate_40 - ret7_gate_30) DESC) AS rank_abs_diff
FROM wide
ORDER BY decile;
