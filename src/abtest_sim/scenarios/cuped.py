"""Сценарий 6. CUPED: меньше дисперсия за счёт метрики до начала теста."""

from .. import generators as gen
from .. import stat_tests as st
from ..runner import Task
from ._common import to_block

NAME = "cuped"


def build_tasks(cfg):
    sc = cfg["scenarios"][NAME]
    return [
        Task(NAME, "pre_period", sc["n"], u, f"rho={rho}", sc["n_reps"], kwargs={"rho": rho})
        for rho in sc["rhos"]
        for u in (0.0, sc["uplift"])
    ]


def simulate(task, n_reps, rng, cfg):
    pp = cfg["metrics"]["pre_period"]
    s = gen.pre_period(
        task.n_per_group, pp["mean"], pp["sd"], task.kwargs["rho"], task.uplift, rng, n_reps
    )
    truth = task.uplift * pp["mean"]
    return [
        to_block("welch_t", st.welch_t(s.y_a, s.y_b), truth),
        to_block("cuped", st.cuped(s.y_a, s.x_a, s.y_b, s.x_b), truth),
    ]
