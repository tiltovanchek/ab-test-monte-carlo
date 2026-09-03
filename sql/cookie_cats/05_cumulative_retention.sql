-- Накопленный retention 7-го дня и z-статистика каждые 1000 игроков (связь с peeking).
-- Времени прихода в данных нет, поэтому порядок по userid условный.
WITH ordered AS (
    SELECT
        version,
        ROW_NUMBER() OVER (PARTITION BY version ORDER BY userid) AS k,
        SUM(retention_7::INT) OVER (
            PARTITION BY version ORDER BY userid ROWS UNBOUNDED PRECEDING
        ) AS retained_cum
    FROM cookie_cats
),
paired AS (
    SELECT
        a.k AS players_per_group,
        a.retained_cum / a.k AS ret7_gate_30,
        b.retained_cum / b.k AS ret7_gate_40,
        (a.retained_cum + b.retained_cum) / (2.0 * a.k) AS pooled
    FROM ordered AS a
    JOIN ordered AS b ON a.k = b.k AND b.version = 'gate_40'
    WHERE a.version = 'gate_30' AND a.k % 1000 = 0
)
SELECT
    players_per_group,
    round(ret7_gate_30, 5) AS ret7_gate_30,
    round(ret7_gate_40, 5) AS ret7_gate_40,
    (ret7_gate_40 - ret7_gate_30) / sqrt(pooled * (1 - pooled) * 2 / players_per_group) AS z,
    p_two_sided((ret7_gate_40 - ret7_gate_30) / sqrt(pooled * (1 - pooled) * 2 / players_per_group)) AS p_value
FROM paired
ORDER BY players_per_group;
