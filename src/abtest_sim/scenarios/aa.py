"""Сценарий 1. A/A-тесты: держит ли критерий 5% ложных срабатываний."""

from .. import generators as gen
from .. import stat_tests as st
from ..runner import Task
from ._common import boot_task, to_block

NAME = "aa"


def build_tasks(cfg):
    sc = cfg["scenarios"][NAME]
    tasks = []
    for metric in ("conversion", "revenue", "ctr", "pre_period"):
        for n in sc["n_grid"]:
            tasks.append(Task(NAME, metric, n, 0.0, n_reps=sc["n_reps"]))
            tasks.append(boot_task(NAME, metric, n, 0.0, "bootstrap", sc["n_reps_bootstrap"]))
    return tasks


def simulate(task, n_reps, rng, cfg):
    m = cfg["metrics"]
    n = task.n_per_group
    boot = task.kwargs.get("boot", False)
    n_boot = cfg["scenarios"][NAME]["n_boot"]

    if task.metric == "conversion":
        if boot:
            a, b = gen.conversion(n, m["conversion"]["p"], 0.0, rng, reps=n_reps)
            return [to_block("bootstrap", st.bootstrap_mean_diff(a, b, rng, n_boot), 0.0)]
        x_a, x_b = gen.conversion_counts(n, m["conversion"]["p"], 0.0, rng, reps=n_reps)
        p_a, p_b = x_a / n, x_b / n
        var_a, var_b = p_a * (1 - p_a) * n / (n - 1), p_b * (1 - p_b) * n / (n - 1)
        return [
            to_block("z_test", st.z_test_counts(x_a, n, x_b, n), 0.0),
            to_block("welch_t", st.welch_from_stats(p_a, var_a, n, p_b, var_b, n), 0.0),
        ]

    if task.metric == "revenue":
        r = m["revenue"]
        a, b = gen.revenue(n, r["p"], r["mu"], r["sigma"], 0.0, rng, reps=n_reps)
        if boot:
            return [to_block("bootstrap", st.bootstrap_mean_diff(a, b, rng, n_boot), 0.0)]
        return [
            to_block("welch_t", st.welch_t(a, b), 0.0),
            to_block("mann_whitney", st.mann_whitney(a, b)),
        ]

    if task.metric == "ctr":
        c = m["ctr"]
        s = gen.ctr(
            n, c["views_mean"], c["views_shape"], c["beta_a"], c["beta_b"], 0.0, rng, n_reps
        )
        if boot:
            return [to_block("bootstrap", st.bootstrap_ratio(*s, rng, n_boot), 0.0)]
        return [
            to_block("naive_t", st.naive_ratio_t(*s), 0.0),
            to_block("delta", st.delta_ratio(*s), 0.0),
        ]

    pp = m["pre_period"]
    s = gen.pre_period(n, pp["mean"], pp["sd"], pp["rho"], 0.0, rng, reps=n_reps)
    if boot:
        return [to_block("bootstrap", st.bootstrap_mean_diff(s.y_a, s.y_b, rng, n_boot), 0.0)]
    return [
        to_block("welch_t", st.welch_t(s.y_a, s.y_b), 0.0),
        to_block("cuped", st.cuped(s.y_a, s.x_a, s.y_b, s.x_b), 0.0),
    ]
