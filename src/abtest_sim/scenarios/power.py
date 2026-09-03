"""Сценарий 2. Мощность z-теста: симуляция против формулы."""

from .. import generators as gen
from .. import stat_tests as st
from ..runner import Task
from ._common import to_block

NAME = "power"


def build_tasks(cfg):
    sc = cfg["scenarios"][NAME]
    return [
        Task(NAME, "conversion", n, u, n_reps=sc["n_reps"])
        for u in sc["uplifts"]
        for n in sc["n_grid"]
    ]


def simulate(task, n_reps, rng, cfg):
    p = cfg["scenarios"][NAME]["p"]
    n = task.n_per_group
    x_a, x_b = gen.conversion_counts(n, p, task.uplift, rng, reps=n_reps)
    return [to_block("z_test", st.z_test_counts(x_a, n, x_b, n), p * task.uplift)]
