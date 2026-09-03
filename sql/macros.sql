-- Макросы. В DuckDB нет функции нормального распределения, поэтому erf считаем
-- приближённо по формуле Абрамовица и Стиган 7.1.26 (ошибка меньше 1.5e-7).

CREATE OR REPLACE MACRO erf_approx(x) AS (
    sign(x) * (1 - (
        0.254829592 * (1 / (1 + 0.3275911 * abs(x)))
        - 0.284496736 * pow(1 / (1 + 0.3275911 * abs(x)), 2)
        + 1.421413741 * pow(1 / (1 + 0.3275911 * abs(x)), 3)
        - 1.453152027 * pow(1 / (1 + 0.3275911 * abs(x)), 4)
        + 1.061405429 * pow(1 / (1 + 0.3275911 * abs(x)), 5)
    ) * exp(-x * x))
);

-- P(Z > z) для N(0, 1)
CREATE OR REPLACE MACRO norm_sf(z) AS 0.5 * (1 - erf_approx(z / sqrt(2)));

-- p-value двустороннего z-теста и хи-квадрата с одной степенью свободы
CREATE OR REPLACE MACRO p_two_sided(z) AS 2 * norm_sf(abs(z));
CREATE OR REPLACE MACRO chi2_df1_p(x) AS 2 * norm_sf(sqrt(x));

-- 95% интервал Уилсона для доли k / n
CREATE OR REPLACE MACRO wilson_lo(k, n) AS (
    (k / n + 1.959964 ^ 2 / (2 * n)
     - 1.959964 * sqrt(k / n * (1 - k / n) / n + 1.959964 ^ 2 / (4 * n * n)))
    / (1 + 1.959964 ^ 2 / n)
);
CREATE OR REPLACE MACRO wilson_hi(k, n) AS (
    (k / n + 1.959964 ^ 2 / (2 * n)
     + 1.959964 * sqrt(k / n * (1 - k / n) / n + 1.959964 ^ 2 / (4 * n * n)))
    / (1 + 1.959964 ^ 2 / n)
);
