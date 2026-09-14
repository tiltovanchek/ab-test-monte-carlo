"""Сценарий 3. Peeking: тест останавливают при первом p < 0.05 на промежуточной проверке."""

import numpy as np
from scipy import stats

from .. import corrections as cr
from ..runner import Block, Task

NAME = "peeking"


def build_tasks(cfg):
    sc = cfg["scenarios"][NAME]
    tasks = []
    for u in (0.0, sc["uplift_power"]):
        for k in sc["looks"]:
            task = Task(
                NAME,
                "conversion",
                sc["n_total"],
                u,
                f"looks={k}",
                sc["n_reps"],
                kwargs={"looks": k},
            )
            tasks.append(task)
    return tasks


def cumulative_z(n_total, looks, p, uplift, rng, reps):
    """z и p-value на каждой из K проверок, форма (reps, K).
    Пользователи приходят порциями по n_total / K, тест считается по всем накопленным."""
    step = n_total // looks
    x_a = rng.binomial(step, p, size=(reps, looks)).cumsum(1)
    x_b = rng.binomial(step, p * (1 + uplift), size=(reps, looks)).cumsum(1)
    n = step * np.arange(1, looks + 1)
    p_pool = (x_a + x_b) / (2 * n)
    se = np.sqrt(p_pool * (1 - p_pool) * 2 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (x_b - x_a) / n / se, 0.0)
    return z, 2 * stats.norm.sf(np.abs(z))


def simulate(task, n_reps, rng, cfg):
    k = task.kwargs["looks"]
    alpha = cfg["alpha"]
    p = cfg["metrics"]["conversion"]["p"]
    z, pv = cumulative_z(task.n_per_group, k, p, task.uplift, rng, n_reps)
    # p_value у всех трёх правил одинаковый (минимум по проверкам),
    # отличается только решение остановить тест - оно в reject
    p_min = pv.min(1)
    pocock = (np.abs(z) > cr.pocock_boundary(k, alpha)).any(1)
    obf = (np.abs(z) > cr.obrien_fleming_boundary(k, alpha)).any(1)
    return [
        Block("naive", p_min, reject=p_min < alpha),
        Block("pocock", p_min, reject=pocock),
        Block("obrien_fleming", p_min, reject=obf),
    ]


def pvalue_trajectories(cfg):
    """p-value после каждой проверки для нескольких A/A-тестов (для графика)."""
    sc = cfg["scenarios"][NAME]
    rng = np.random.default_rng(cfg["seed"])
    p = cfg["metrics"]["conversion"]["p"]
    _, pv = cumulative_z(sc["n_total"], sc["trajectory_looks"], p, 0.0, rng, sc["trajectories"])
    return pv
