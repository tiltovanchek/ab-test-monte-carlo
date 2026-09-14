"""Статистические критерии на NumPy.

Критерии считают по последней оси: можно передать один эксперимент (массивы длины n)
или сразу пачку (матрицы reps x n), тогда всё считается без цикла.
Все возвращают TestResult(p_value, effect, ci_low, ci_high), effect - разница B - A.
Все тесты двусторонние.
"""

from typing import NamedTuple

import numpy as np
from scipy import stats

ALPHA = 0.05


class TestResult(NamedTuple):
    p_value: np.ndarray
    effect: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray


def _z(alpha):
    return stats.norm.ppf(1 - alpha / 2)


def _no_ci(effect):
    nan = np.full(np.shape(effect), np.nan)
    return nan, nan


def _normal_result(diff, se, alpha):
    p_value = 2 * stats.norm.sf(np.abs(diff / se))
    return TestResult(p_value, diff, diff - _z(alpha) * se, diff + _z(alpha) * se)


def z_test_counts(x_a, n_a, x_b, n_b, alpha=ALPHA):
    """Z-тест для двух долей по числу успехов x и размерам групп n.
    В статистике объединённая доля, в интервале - дисперсии групп по отдельности."""
    x_a, n_a, x_b, n_b = (np.asarray(v, dtype=float) for v in (x_a, n_a, x_b, n_b))
    p_a, p_b = x_a / n_a, x_b / n_b
    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    diff = p_b - p_a
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se_pool > 0, diff / se_pool, 0.0)
    se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    p_value = 2 * stats.norm.sf(np.abs(z))
    return TestResult(p_value, diff, diff - _z(alpha) * se, diff + _z(alpha) * se)


def z_test_proportions(a, b, alpha=ALPHA):
    """Z-тест для долей по массивам из 0 и 1."""
    return z_test_counts(a.sum(-1), a.shape[-1], b.sum(-1), b.shape[-1], alpha)


def welch_from_stats(mean_a, var_a, n_a, mean_b, var_b, n_b, alpha=ALPHA):
    """t-тест Уэлча по средним и дисперсиям групп."""
    va, vb = var_a / n_a, var_b / n_b
    se = np.sqrt(va + vb)
    diff = mean_b - mean_a
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, diff / se, 0.0)
        df = (va + vb) ** 2 / (va**2 / (n_a - 1) + vb**2 / (n_b - 1))
    df = np.where(np.isfinite(df), df, 1.0)
    p_value = 2 * stats.t.sf(np.abs(t), df)
    tc = stats.t.ppf(1 - alpha / 2, df)
    return TestResult(p_value, diff, diff - tc * se, diff + tc * se)


def welch_t(a, b, alpha=ALPHA):
    """t-тест Уэлча (дисперсии групп могут быть разными)."""
    mean_a, var_a = a.mean(-1), a.var(-1, ddof=1)
    mean_b, var_b = b.mean(-1), b.var(-1, ddof=1)
    return welch_from_stats(mean_a, var_a, a.shape[-1], mean_b, var_b, b.shape[-1], alpha)


def log_t(a, b, alpha=ALPHA):
    """t-тест на log(1 + x). Сравнивает уже не средние, а "типичные" значения."""
    return welch_t(np.log1p(a), np.log1p(b), alpha)


def winsorized_t(a, b, q=0.99, alpha=ALPHA):
    """t-тест после отсечения значений выше q-квантиля обеих групп вместе."""
    cap = np.quantile(np.concatenate([a, b], axis=-1), q, axis=-1, keepdims=True)
    return welch_t(np.minimum(a, cap), np.minimum(b, cap), alpha)


def mann_whitney(a, b, alpha=ALPHA):
    """Манн-Уитни, нормальное приближение с поправкой на одинаковые значения.
    effect - доля пар, где значение из B больше значения из A (без эффекта около 0.5)."""
    n_a, n_b = a.shape[-1], b.shape[-1]
    n = n_a + n_b
    ranks = stats.rankdata(np.concatenate([a, b], axis=-1), axis=-1)
    u_b = ranks[..., n_a:].sum(-1) - n_b * (n_b + 1) / 2

    # дисперсия U через сумму квадратов рангов - так сразу учитываются одинаковые значения
    sum_sq = (ranks**2).sum(-1)
    var_u = n_a * n_b * (sum_sq / (n * (n - 1)) - (n + 1) ** 2 / (4 * (n - 1)))
    sigma = np.sqrt(np.maximum(var_u, 0))

    # как в scipy: берём больший из двух U и поправку на непрерывность 0.5
    u = np.maximum(u_b, n_a * n_b - u_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, (u - n_a * n_b / 2 - 0.5) / sigma, 0.0)
    p_value = np.minimum(1.0, 2 * stats.norm.sf(z))
    effect = u_b / (n_a * n_b)
    return TestResult(p_value, effect, *_no_ci(effect))


def _bootstrap_result(diffs, effect, alpha):
    """Перцентильный интервал и p-value, согласованный с ним:
    p < alpha ровно тогда, когда 0 не попал в интервал."""
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p_value = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return TestResult(p_value, effect, lo, hi)


def _stack(results):
    return TestResult(*(np.array(v) for v in zip(*results, strict=True)))


def bootstrap_mean_diff(a, b, rng, n_boot=1000, alpha=ALPHA):
    """Бутстрап разницы средних. Пачку считаем по одному эксперименту,
    иначе индексы reps x n_boot x n не влезут в память."""
    if a.ndim == 2:
        return _stack(
            bootstrap_mean_diff(x, y, rng, n_boot, alpha) for x, y in zip(a, b, strict=True)
        )

    def boot_means(x):
        idx = rng.integers(0, x.size, size=(n_boot, x.size), dtype=np.int32)
        return x[idx].mean(-1)

    diffs = boot_means(b) - boot_means(a)
    return _bootstrap_result(diffs, b.mean() - a.mean(), alpha)


def naive_ratio_t(clicks_a, views_a, clicks_b, views_b, alpha=ALPHA):
    """Наивный t-тест: каждый просмотр считается отдельным наблюдением 0/1.
    Это специально неверный вариант: просмотры одного пользователя зависимы,
    а тест этого не видит и занижает дисперсию."""

    def group(clicks, views):
        n_views = views.sum(-1)
        p = clicks.sum(-1) / n_views
        return p, p * (1 - p) * n_views / (n_views - 1), n_views

    return welch_from_stats(*group(clicks_a, views_a), *group(clicks_b, views_b), alpha)


def _delta(clicks, views):
    """Отношение средних и его дисперсия по дельта-методу."""
    n = clicks.shape[-1]
    mean_c, mean_v = clicks.mean(-1), views.mean(-1)
    var_c, var_v = clicks.var(-1, ddof=1), views.var(-1, ddof=1)
    cov = ((clicks - mean_c[..., None]) * (views - mean_v[..., None])).sum(-1) / (n - 1)
    r = mean_c / mean_v
    var_r = (var_c - 2 * r * cov + r**2 * var_v) / (n * mean_v**2)
    return r, var_r


def delta_ratio(clicks_a, views_a, clicks_b, views_b, alpha=ALPHA):
    """Z-тест для разницы метрик-отношений, дисперсия по дельта-методу."""
    r_a, var_a = _delta(clicks_a, views_a)
    r_b, var_b = _delta(clicks_b, views_b)
    return _normal_result(r_b - r_a, np.sqrt(var_a + var_b), alpha)


def bootstrap_ratio(clicks_a, views_a, clicks_b, views_b, rng, n_boot=1000, alpha=ALPHA):
    """Бутстрап по пользователям для разницы метрик-отношений."""
    if clicks_a.ndim == 2:
        rows = zip(clicks_a, views_a, clicks_b, views_b, strict=True)
        return _stack(bootstrap_ratio(*row, rng, n_boot, alpha) for row in rows)

    def boot_ratios(clicks, views):
        idx = rng.integers(0, clicks.size, size=(n_boot, clicks.size), dtype=np.int32)
        return clicks[idx].sum(-1) / views[idx].sum(-1)

    diffs = boot_ratios(clicks_b, views_b) - boot_ratios(clicks_a, views_a)
    effect = clicks_b.sum() / views_b.sum() - clicks_a.sum() / views_a.sum()
    return _bootstrap_result(diffs, effect, alpha)


def cuped_theta(x, y):
    """theta = cov(X, Y) / var(X)."""
    xc = x - x.mean(-1, keepdims=True)
    yc = y - y.mean(-1, keepdims=True)
    return (xc * yc).sum(-1) / (xc**2).sum(-1)


def cuped(y_a, x_a, y_b, x_b, alpha=ALPHA):
    """t-тест на Y - theta * (X - mean(X)). theta и mean(X) считаем по обеим
    группам вместе, поэтому поправка одинаковая и оценка эффекта не смещается."""
    x = np.concatenate([x_a, x_b], axis=-1)
    y = np.concatenate([y_a, y_b], axis=-1)
    theta = cuped_theta(x, y)[..., None]
    x_mean = x.mean(-1, keepdims=True)
    return welch_t(y_a - theta * (x_a - x_mean), y_b - theta * (x_b - x_mean), alpha)


def srm_chi2(n_a, n_b, expected_share_b=0.5):
    """Хи-квадрат: совпадает ли разбиение на группы с ожидаемым.
    effect - насколько доля B отличается от ожидаемой."""
    n_a = np.asarray(n_a, dtype=float)
    n_b = np.asarray(n_b, dtype=float)
    total = n_a + n_b
    e_a, e_b = total * (1 - expected_share_b), total * expected_share_b
    chi2 = (n_a - e_a) ** 2 / e_a + (n_b - e_b) ** 2 / e_b
    effect = n_b / total - expected_share_b
    return TestResult(stats.chi2.sf(chi2, df=1), effect, *_no_ci(effect))


def wilson_ci(successes, n, alpha=ALPHA):
    """Интервал Уилсона для доли: FPR, мощности и т.п."""
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    z = _z(alpha)
    p = successes / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return center - half, center + half
