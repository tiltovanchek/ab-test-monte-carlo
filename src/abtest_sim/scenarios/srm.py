"""Сценарий 8. SRM: ловит ли хи-квадрат сломанное разбиение на группы."""

import numpy as np

from .. import generators as gen
from .. import stat_tests as st
from ..runner import Block, Task

NAME = "srm"


def build_tasks(cfg):
    sc = cfg["scenarios"][NAME]
    tasks = []
    for total in sc["n_totals"]:
        for share in sc["shares_b"]:
            param = f"N={total};share_b={share}"
            kw = {"total": total, "share_b": share}
            # массивов наблюдений тут нет, поэтому чанк можно взять большим
            task = Task(
                NAME,
                "split",
                total // 2,
                0.0,
                param,
                sc["n_reps"],
                alpha=sc["alpha_srm"],
                chunk_size=1000,
                kwargs=kw,
            )
            tasks.append(task)

    d = sc["dropout"]
    for rate in d["drop_rates"]:
        kw = {"drop_rate": rate}
        tasks.append(
            Task(NAME, "activity", d["n_total"] // 2, 0.0, f"drop={rate}", sc["n_reps"], kwargs=kw)
        )
    return tasks


def simulate(task, n_reps, rng, cfg):
    sc = cfg["scenarios"][NAME]
    if task.metric == "split":
        n_a, n_b = gen.split(task.kwargs["total"], task.kwargs["share_b"], rng, reps=n_reps)
        res = st.srm_chi2(n_a, n_b)
        return [Block("chi2", res.p_value, effect=res.effect)]

    # A/A-тест с выпадением пользователей из B. Размер B в каждом эксперименте свой,
    # поэтому тут обычный цикл
    quantile = sc["dropout"]["low_activity_quantile"]
    p_t, eff_t, p_srm = np.empty(n_reps), np.empty(n_reps), np.empty(n_reps)
    for i in range(n_reps):
        a, b = gen.activity_with_dropout(task.n_per_group, task.kwargs["drop_rate"], quantile, rng)
        t = st.welch_t(a, b)
        p_t[i], eff_t[i] = t.p_value, t.effect
        p_srm[i] = st.srm_chi2(a.size, b.size).p_value
    return [
        Block("welch_t", p_t, reject=p_t < cfg["alpha"], effect=eff_t),
        Block("chi2", p_srm, reject=p_srm < sc["alpha_srm"]),
    ]
