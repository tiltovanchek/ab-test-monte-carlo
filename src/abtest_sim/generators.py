"""Генераторы метрик с известным истинным эффектом.

Все генераторы принимают размер группы n, параметры, относительный эффект uplift
(0 - это A/A-тест) и rng. Если передать reps, вернутся матрицы reps x n,
то есть сразу пачка независимых экспериментов.
"""

from typing import NamedTuple

import numpy as np
from scipy.stats import norm


def _shape(n, reps):
    return (n,) if reps is None else (reps, n)


def conversion(n, p, uplift, rng, reps=None):
    """Конверсия: в A вероятность p, в B p * (1 + uplift)."""
    a = (rng.random(_shape(n, reps)) < p).astype(float)
    b = (rng.random(_shape(n, reps)) < p * (1 + uplift)).astype(float)
    return a, b


def conversion_counts(n, p, uplift, rng, reps):
    """Сразу число конверсий в группе, без массива наблюдений. Так намного быстрее."""
    return rng.binomial(n, p, size=reps), rng.binomial(n, p * (1 + uplift), size=reps)


def _revenue(shape, p, mu, sigma, rng):
    paid = rng.random(shape) < p
    check = rng.lognormal(mu, sigma, size=shape)
    return np.where(paid, check, 0.0)


def revenue(n, p, mu, sigma, uplift, rng, reps=None):
    """Выручка на пользователя: конверсия p и логнормальный чек.
    Эффект - средний чек в B больше в (1 + uplift) раз."""
    shape = _shape(n, reps)
    a = _revenue(shape, p, mu, sigma, rng)
    b = _revenue(shape, p, mu + np.log1p(uplift), sigma, rng)
    return a, b


def revenue_mean(p, mu, sigma):
    return p * np.exp(mu + sigma**2 / 2)


def revenue_shape_shift(n, p, mu, sigma_a, sigma_b, rng, reps=None):
    """Средняя выручка в группах одинаковая, а форма распределения разная:
    в B больше sigma, поэтому больше мелких чеков и изредка очень крупные."""
    shape = _shape(n, reps)
    mu_b = mu + sigma_a**2 / 2 - sigma_b**2 / 2
    a = _revenue(shape, p, mu, sigma_a, rng)
    b = _revenue(shape, p, mu_b, sigma_b, rng)
    return a, b


class RatioSample(NamedTuple):
    clicks_a: np.ndarray
    views_a: np.ndarray
    clicks_b: np.ndarray
    views_b: np.ndarray


def _ctr(shape, views_mean, views_shape, beta_a, beta_b, uplift, rng):
    lam = rng.gamma(views_shape, views_mean / views_shape, size=shape)
    views = rng.poisson(lam)
    p = np.clip(rng.beta(beta_a, beta_b, size=shape) * (1 + uplift), 0, 1)
    clicks = rng.binomial(views, p)
    return clicks.astype(float), views.astype(float)


def ctr(n, views_mean, views_shape, beta_a, beta_b, uplift, rng, reps=None):
    """CTR: у каждого пользователя свои просмотры (Пуассон с Гамма-интенсивностью)
    и своя вероятность клика (Бета). Эффект - вероятность клика выше в (1 + uplift) раз."""
    shape = _shape(n, reps)
    ca, va = _ctr(shape, views_mean, views_shape, beta_a, beta_b, 0.0, rng)
    cb, vb = _ctr(shape, views_mean, views_shape, beta_a, beta_b, uplift, rng)
    return RatioSample(ca, va, cb, vb)


def beta_params(mean, concentration):
    """Параметры a, b Бета-распределения по среднему и сумме a + b."""
    return mean * concentration, (1 - mean) * concentration


class PrePeriodSample(NamedTuple):
    x_a: np.ndarray
    y_a: np.ndarray
    x_b: np.ndarray
    y_b: np.ndarray


def pre_period(n, mean, sd, rho, uplift, rng, reps=None):
    """X - метрика до теста, Y - во время теста, корреляция rho.
    В группе B к Y добавляется uplift * mean."""
    shape = _shape(n, reps)

    def group(shift):
        z1 = rng.standard_normal(shape)
        z2 = rng.standard_normal(shape)
        x = mean + sd * z1
        y = mean + shift + sd * (rho * z1 + np.sqrt(1 - rho**2) * z2)
        return x, y

    x_a, y_a = group(0.0)
    x_b, y_b = group(uplift * mean)
    return PrePeriodSample(x_a, y_a, x_b, y_b)


def multi_metric(n, k, rho, effects, rng, reps=None):
    """k метрик N(0, 1) с попарной корреляцией rho, форма (reps, n, k).
    effects - сдвиг каждой метрики в B в стандартных отклонениях."""
    cov = np.full((k, k), rho) + (1 - rho) * np.eye(k)
    chol = np.linalg.cholesky(cov)
    shape = (*_shape(n, reps), k)
    a = rng.standard_normal(shape) @ chol.T
    b = rng.standard_normal(shape) @ chol.T + effects
    return a, b


def split(n_total, share_b, rng, reps=None):
    """Сколько пользователей попало в A и B, если доля B равна share_b."""
    n_b = rng.binomial(n_total, share_b, size=reps)
    return n_total - n_b, n_b


def activity_with_dropout(n, drop_rate, low_quantile, rng, mu=2.0, sigma=1.0):
    """A/A-тест, где из B пропадает drop_rate * n малоактивных пользователей
    (из нижних low_quantile по активности)."""
    a = rng.lognormal(mu, sigma, size=n)
    b = rng.lognormal(mu, sigma, size=n)
    n_drop = round(drop_rate * n)
    if n_drop:
        threshold = np.exp(mu + sigma * norm.ppf(low_quantile))
        low = np.flatnonzero(b < threshold)
        b = np.delete(b, rng.choice(low, size=min(n_drop, low.size), replace=False))
    return a, b
