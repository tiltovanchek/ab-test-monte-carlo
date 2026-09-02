"""Формулы мощности и размера выборки."""

import math

import numpy as np
from scipy import stats


def required_sample_size(p, uplift, alpha=0.05, power=0.80):
    """Размер каждой группы для z-теста двух долей:
    n = (z(1 - alpha/2) + z(power))^2 * (pA(1 - pA) + pB(1 - pB)) / (pB - pA)^2,
    где pB = p * (1 + uplift)."""
    p_b = p * (1 + uplift)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    n = z**2 * (p * (1 - p) + p_b * (1 - p_b)) / (p_b - p) ** 2
    return math.ceil(n)


def power_two_proportions(p, uplift, n, alpha=0.05):
    """Теоретическая мощность z-теста двух долей при n пользователей в группе."""
    p_b = p * (1 + uplift)
    se = np.sqrt((p * (1 - p) + p_b * (1 - p_b)) / np.asarray(n, dtype=float))
    shift = abs(p_b - p) / se
    z = stats.norm.ppf(1 - alpha / 2)
    return stats.norm.sf(z - shift) + stats.norm.cdf(-z - shift)


def cuped_variance_ratio(rho):
    """Какая доля дисперсии (и нужной выборки) остаётся после CUPED: 1 - rho^2."""
    return 1 - np.asarray(rho) ** 2
