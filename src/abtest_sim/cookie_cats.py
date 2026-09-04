"""Итоговые тесты по Cookie Cats и A/A-проверка на перемешанных метках групп."""

import numpy as np
import pandas as pd

from . import db
from . import stat_tests as st


def load():
    """Игроки из sql/cookie_cats/06_export.sql."""
    return db.query("cookie_cats/06_export")


def final_tests(df, rng, n_boot=5000):
    """z-тест и бутстрап для retention, t-тест и Манн-Уитни для числа раундов."""
    a, b = df[df["version"] == "gate_30"], df[df["version"] == "gate_40"]
    rows = []
    for metric in ("retention_1", "retention_7"):
        x, y = a[metric].to_numpy(float), b[metric].to_numpy(float)
        rows.append((metric, "z_test", st.z_test_proportions(x, y)))
        rows.append((metric, "bootstrap", st.bootstrap_mean_diff(x, y, rng, n_boot)))
    x, y = a["sum_gamerounds"].to_numpy(float), b["sum_gamerounds"].to_numpy(float)
    rows.append(("sum_gamerounds", "welch_t", st.welch_t(x, y)))
    rows.append(("sum_gamerounds", "mann_whitney", st.mann_whitney(x, y)))

    table = [(metric, test, *map(float, res)) for metric, test, res in rows]
    return pd.DataFrame(table, columns=["metric", "test", "p_value", "effect", "ci_low", "ci_high"])


def shuffled_aa(df, rng, n_reps=2000):
    """Перемешиваем метки групп: настоящего эффекта нет, доля p < 0.05 должна быть около 5%."""
    n_b = (df["version"] == "gate_40").sum()
    tests = {
        "retention_1": st.z_test_proportions,
        "retention_7": st.z_test_proportions,
        "sum_gamerounds": st.welch_t,
    }
    values = {metric: df[metric].to_numpy(float) for metric in tests}
    rejects = dict.fromkeys(tests, 0)
    for _ in range(n_reps):
        is_b = np.zeros(len(df), dtype=bool)
        is_b[rng.choice(len(df), size=n_b, replace=False)] = True
        for metric, test in tests.items():
            x = values[metric]
            rejects[metric] += test(x[~is_b], x[is_b]).p_value < 0.05

    rows = []
    for metric, test in tests.items():
        lo, hi = st.wilson_ci(rejects[metric], n_reps)
        rows.append((metric, test.__name__, n_reps, rejects[metric] / n_reps, float(lo), float(hi)))
    return pd.DataFrame(rows, columns=["metric", "test", "reps", "fpr", "ci_low", "ci_high"])
