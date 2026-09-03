-- Размеры групп и проверка SRM: хи-квадрат против 50/50 прямо в SQL, порог p < 0.001.
WITH sizes AS (
    SELECT version, count(*) AS players
    FROM cookie_cats
    GROUP BY version
),
expected AS (
    SELECT *, sum(players) OVER () / 2.0 AS expected_players
    FROM sizes
),
chi AS (
    SELECT sum((players - expected_players) ^ 2 / expected_players) AS chi2
    FROM expected
)
SELECT
    e.version,
    e.players,
    round(e.players / sum(e.players) OVER (), 5) AS share,
    round(chi.chi2, 4)                            AS chi2,
    chi2_df1_p(chi.chi2)                          AS p_value,
    chi2_df1_p(chi.chi2) < 0.001                  AS srm_detected
FROM expected AS e, chi
ORDER BY e.version;
