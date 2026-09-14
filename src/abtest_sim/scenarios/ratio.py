"""Сценарий 5. CTR: наивный тест по просмотрам, дельта-метод и бутстрап по пользователям."""

from .. import generators as gen
from .. import stat_tests as st
from ..runner import Task
from ._common import boot_task, to_block

NAME = "ratio"


def build_tasks(cfg):
    sc = cfg["scenarios"][NAME]
    n = sc["n"]
    tasks = []
    for level, conc in sc["heterogeneity"].items():
        for u in sc["uplifts"]:
            param = f"heterogeneity={level}"
            boot_param, boot_reps = param + ";bootstrap", sc["n_reps_bootstrap"]
            tasks.append(
                Task(NAME, "ctr", n, u, param, sc["n_reps"], kwargs={"concentration": conc})
            )
            tasks.append(boot_task(NAME, "ctr", n, u, boot_param, boot_reps, concentration=conc))
    return tasks


def simulate(task, n_reps, rng, cfg):
    c = cfg["metrics"]["ctr"]
    mean_ctr = c["beta_a"] / (c["beta_a"] + c["beta_b"])
    beta_a, beta_b = gen.beta_params(mean_ctr, task.kwargs["concentration"])
    views = c["views_mean"], c["views_shape"]
    s = gen.ctr(task.n_per_group, *views, beta_a, beta_b, task.uplift, rng, n_reps)
    # просмотры и вероятность клика независимы, поэтому истинный CTR группы - среднее p
    truth = mean_ctr * task.uplift
    if task.kwargs.get("boot"):
        n_boot = cfg["scenarios"][NAME]["n_boot"]
        return [to_block("bootstrap", st.bootstrap_ratio(*s, rng, n_boot), truth)]
    return [
        to_block("naive_t", st.naive_ratio_t(*s), truth),
        to_block("delta", st.delta_ratio(*s), truth),
    ]
