"""Сценарий 7. Множественное тестирование.

Чтобы не хранить строку на каждую метрику, метрики сворачиваются в две группы:
metric = "null" (без эффекта) и "true" (с эффектом). В строке:
reject - была ли в группе хоть одна значимая метрика, effect - сколько их было,
p_value - минимальное скорректированное p, uplift - сдвиг метрик группы в sd.
"""

import numpy as np

from .. import corrections as cr
from .. import generators as gen
from .. import stat_tests as st
from ..runner import Block, Task

NAME = "multiple"


def _task(n, uplift, k, rho, n_true, n_reps):
    param = f"k={k};rho={rho};true={n_true}"
    return Task(
        NAME, "normal", n, uplift, param, n_reps, kwargs={"k": k, "rho": rho, "n_true": n_true}
    )


def build_tasks(cfg):
    sc = cfg["scenarios"][NAME]
    we = sc["with_effect"]
    tasks = []
    for rho in sc["rhos"]:
        tasks += [_task(sc["n"], 0.0, k, rho, 0, sc["n_reps"]) for k in sc["ks"]]
        tasks.append(_task(sc["n"], we["effect_sd"], we["k"], rho, we["n_true"], sc["n_reps"]))
    return tasks


def simulate(task, n_reps, rng, cfg):
    k, rho, n_true = task.kwargs["k"], task.kwargs["rho"], task.kwargs["n_true"]
    n = task.n_per_group
    effects = np.zeros(k)
    effects[:n_true] = task.uplift

    # массив reps x n x k большой, поэтому считаем порциями
    step = max(1, cfg["max_chunk_cells"] // (n * k))
    p = []
    for start in range(0, n_reps, step):
        a, b = gen.multi_metric(n, k, rho, effects, rng, reps=min(step, n_reps - start))
        # тест считается по последней оси, поэтому переставляем оси в (reps, k, n)
        p.append(st.welch_t(a.swapaxes(1, 2), b.swapaxes(1, 2)).p_value)
    p = np.concatenate(p)

    groups = [("null", slice(n_true, k), 0.0)]
    if n_true:
        groups.append(("true", slice(0, n_true), task.uplift))

    blocks = []
    for name, correct in cr.CORRECTIONS.items():
        adj = correct(p)
        for metric, cols, uplift in groups:
            rejected = adj[:, cols] < task.alpha
            p_min = adj[:, cols].min(1)
            block = Block(
                name, p_min, rejected.any(1), rejected.sum(1), metric=metric, uplift=uplift
            )
            blocks.append(block)
    return blocks
