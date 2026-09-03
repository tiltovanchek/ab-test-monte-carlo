"""Сценарий 4. Тяжёлые хвосты: выручка с логнормальным чеком разной sigma."""

from .. import generators as gen
from .. import stat_tests as st
from ..runner import Task
from ._common import boot_task, to_block

NAME = "heavy_tails"


def build_tasks(cfg):
    sc = cfg["scenarios"][NAME]
    n, reps, boot_reps = sc["n"], sc["n_reps"], sc["n_reps_bootstrap"]
    tasks = []
    for sigma in sc["sigmas"]:
        for u in (0.0, sc["uplift"]):
            param = f"sigma={sigma}"
            tasks.append(
                Task(NAME, "revenue", n, u, param=param, n_reps=reps, kwargs={"sigma": sigma})
            )
            tasks.append(
                boot_task(NAME, "revenue", n, u, param + ";bootstrap", boot_reps, sigma=sigma)
            )

    trap = sc["shape_trap"]
    param = f"sigma_a={trap['sigma_a']};sigma_b={trap['sigma_b']}"
    tasks.append(Task(NAME, "revenue_shape", n, 0.0, param=param, n_reps=reps))
    tasks.append(boot_task(NAME, "revenue_shape", n, 0.0, param + ";bootstrap", boot_reps))
    return tasks


def simulate(task, n_reps, rng, cfg):
    sc = cfg["scenarios"][NAME]
    r = cfg["metrics"]["revenue"]
    n = task.n_per_group

    if task.metric == "revenue":
        sigma = task.kwargs["sigma"]
        a, b = gen.revenue(n, r["p"], r["mu"], sigma, task.uplift, rng, reps=n_reps)
        truth = gen.revenue_mean(r["p"], r["mu"], sigma) * task.uplift
    else:
        trap = sc["shape_trap"]
        a, b = gen.revenue_shape_shift(
            n, r["p"], r["mu"], trap["sigma_a"], trap["sigma_b"], rng, n_reps
        )
        truth = 0.0

    if task.kwargs.get("boot"):
        return [to_block("bootstrap", st.bootstrap_mean_diff(a, b, rng, sc["n_boot"]), truth)]
    return [
        to_block("welch_t", st.welch_t(a, b), truth),
        to_block("mann_whitney", st.mann_whitney(a, b)),
        to_block("log_t", st.log_t(a, b)),
        to_block("winsorized_t", st.winsorized_t(a, b, sc["trim_quantile"])),
    ]
