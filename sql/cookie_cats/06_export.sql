-- Выгрузка игроков для итоговых тестов в Python.
SELECT
    version,
    retention_1::INT AS retention_1,
    retention_7::INT AS retention_7,
    sum_gamerounds
FROM cookie_cats
ORDER BY userid;
