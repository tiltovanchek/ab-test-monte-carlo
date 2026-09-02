"""Поправки на множественные сравнения и границы для подглядывания (peeking)."""

from functools import lru_cache

import numpy as np
from scipy import optimize, signal, stats

# Поправки принимают p-value формы (..., k) и возвращают скорректированные p-value.
# Гипотеза отвергается, если скорректированное p < alpha.


def bonferroni(p):
    """p * k. Держит FWER при любой зависимости между метриками."""
    return np.minimum(1.0, p * p.shape[-1])


def holm(p):
    """Пошаговый Бонферрони: тоже держит FWER, но всегда не слабее Бонферрони."""
    k = p.shape[-1]
    order = np.argsort(p, axis=-1)
    p_sorted = np.take_along_axis(p, order, axis=-1)
    adj = np.maximum.accumulate(p_sorted * (k - np.arange(k)), axis=-1)
    return _unsort(np.minimum(1.0, adj), order)


def benjamini_hochberg(p):
    """Держит FDR (долю ложных среди найденных), а не FWER."""
    k = p.shape[-1]
    order = np.argsort(p, axis=-1)
    p_sorted = np.take_along_axis(p, order, axis=-1)
    adj = p_sorted * k / np.arange(1, k + 1)
    adj = np.minimum.accumulate(adj[..., ::-1], axis=-1)[..., ::-1]
    return _unsort(np.minimum(1.0, adj), order)


def _unsort(values, order):
    out = np.empty_like(values)
    np.put_along_axis(out, order, values, axis=-1)
    return out


CORRECTIONS = {
    "none": lambda p: p,
    "bonferroni": bonferroni,
    "holm": holm,
    "bh": benjamini_hochberg,
}


def crossing_probability(z_bounds, step=0.01):
    """Вероятность при H0 хотя бы раз выйти за границы |Z_k| > z_k.

    Проверки идут через равные промежутки: S_k - сумма k независимых N(0, 1),
    Z_k = S_k / sqrt(k). Плотность S считаем на сетке: от проверки к проверке
    сворачиваем её с N(0, 1) и обнуляем за границей.
    """
    z_bounds = np.asarray(z_bounds, dtype=float)
    s_bounds = z_bounds * np.sqrt(np.arange(1, z_bounds.size + 1))
    half = s_bounds.max() + 8
    grid = np.arange(-half, half + step / 2, step)
    kernel = stats.norm.pdf(np.arange(-8, 8 + step / 2, step)) * step

    density = stats.norm.pdf(grid)
    for i, b in enumerate(s_bounds):
        if i > 0:
            density = signal.fftconvolve(density, kernel, mode="same")
        density = np.where(np.abs(grid) < b, density, 0.0)
    return float(1 - density.sum() * step)


@lru_cache
def pocock_boundary(n_looks, alpha=0.05):
    """Граница Покока: одна и та же |Z| > c на каждой из K проверок."""
    if n_looks == 1:
        return float(stats.norm.ppf(1 - alpha / 2))
    return optimize.brentq(
        lambda c: crossing_probability(np.full(n_looks, c)) - alpha, 1.9, 5.0, xtol=1e-5
    )


@lru_cache
def obrien_fleming_constant(n_looks, alpha=0.05):
    """Константа C в границе О'Брайена-Флеминга z_k = C * sqrt(K / k)."""
    if n_looks == 1:
        return float(stats.norm.ppf(1 - alpha / 2))
    k = np.arange(1, n_looks + 1)
    return optimize.brentq(
        lambda c: crossing_probability(c * np.sqrt(n_looks / k)) - alpha, 1.5, 5.0, xtol=1e-5
    )


def obrien_fleming_boundary(n_looks, alpha=0.05):
    """Границы О'Брайена-Флеминга для каждой проверки: строгие в начале, мягкие в конце."""
    k = np.arange(1, n_looks + 1)
    return obrien_fleming_constant(n_looks, alpha) * np.sqrt(n_looks / k)
