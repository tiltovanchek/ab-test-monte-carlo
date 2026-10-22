"""Графики отчёта. Данные берутся SQL-запросами из sql/simulations.

Правила: заголовок графика - это вывод, у каждой оценки из симуляции
есть 95% интервал, цвета серий всегда в одном порядке.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from . import db  # noqa: E402
from .config import load_config, path_in_root  # noqa: E402
from .power import cuped_variance_ratio, power_two_proportions  # noqa: E402
from .scenarios.peeking import pvalue_trajectories as _trajectories  # noqa: E402
from .stat_tests import wilson_ci  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
BAND = 0.0043  # полуширина 95% интервала для FPR 5% при R = 10 000

TEST_NAMES = {
    "z_test": "z-тест",
    "welch_t": "t-тест Уэлча",
    "mann_whitney": "Манн-Уитни",
    "bootstrap": "бутстрап",
    "log_t": "t-тест на log",
    "winsorized_t": "t-тест с отсечением 99%",
    "naive_t": "наивный t-тест по просмотрам",
    "delta": "дельта-метод",
    "cuped": "CUPED",
    "naive": "без поправки",
    "pocock": "граница Покока",
    "obrien_fleming": "граница О'Брайена-Флеминга",
    "none": "без поправки",
    "bonferroni": "Бонферрони",
    "holm": "Холм",
    "bh": "Беньямини-Хохберг",
}
METRIC_NAMES = {
    "conversion": "Конверсия",
    "revenue": "Выручка на пользователя",
    "ctr": "CTR (клики / просмотры)",
    "pre_period": "Метрика с предпериодом",
}
LOOKS = [1, 2, 5, 10, 20, 50]


def setup_style():
    plt.rcParams.update(
        {
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.edgecolor": MUTED,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "font.size": 10,
        }
    )


def _pct(ax):
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def _alpha_line(ax, band=True):
    """Линия 5% и серая полоса, в которую должен попадать корректный тест."""
    ax.axhline(0.05, color=MUTED, lw=1, ls="--", zorder=1)
    if band:
        ax.axhspan(0.05 - BAND, 0.05 + BAND, color=GRID, alpha=0.7, zorder=0)


def _points(ax, x, rate, lo, hi, color, label, line=True):
    """Оценки с усами 95% интервала."""
    style = "-" if line else "none"
    yerr = [rate - lo, hi - rate]
    ax.errorbar(x, rate, yerr, marker="o", ms=5, capsize=2, color=color, label=label, ls=style)


def _looks_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(LOOKS, labels=[str(k) for k in LOOKS])
    ax.set_xlabel("Число промежуточных проверок K")


def _suptitle(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, fontweight="bold")


def _save(fig, name):
    path = path_in_root(load_config()["figures_dir"]) / f"{name}.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _rates(con):
    return db.query("simulations/01_reject_rates", con)


def peeking_fpr(con):
    df = _rates(con).query("scenario == 'peeking'").copy()
    df["looks"] = df["param"].str.extract(r"looks=(\d+)")[0].astype(int)
    df = df.sort_values("looks")
    h0, h1 = df[df["uplift"] == 0], df[df["uplift"] > 0]
    naive10 = h0.query("test == 'naive' and looks == 10")["rate"].iloc[0]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    for color, test in zip(COLORS, ["naive", "pocock", "obrien_fleming"], strict=False):
        d = h0[h0["test"] == test]
        ax.plot(d["looks"], d["rate"], marker="o", ms=5, color=color, label=TEST_NAMES[test])
        ax.fill_between(d["looks"], d["ci_low"], d["ci_high"], color=color, alpha=0.2, lw=0)
        d = h1[h1["test"] == test]
        _points(ax2, d["looks"], d["rate"], d["ci_low"], d["ci_high"], color, TEST_NAMES[test])

    _alpha_line(ax)
    _looks_axis(ax)
    ax.set_ylabel("Доля ложных срабатываний (A/A)")
    _pct(ax)
    ax.legend(loc="upper left")
    ax.set_title(f"Подглядывание 10 раз поднимает FPR с 5% до {naive10:.0%}")

    _looks_axis(ax2)
    ax2.set_ylabel(f"Мощность при uplift +{h1['uplift'].iloc[0]:.0%}")
    _pct(ax2)
    ax2.set_title("Цена исправления: потеря мощности", fontsize=11)
    fig.tight_layout()
    return _save(fig, "peeking_fpr")


def pvalue_trajectories():
    cfg = load_config()
    pv = _trajectories(cfg)
    n_total = cfg["scenarios"]["peeking"]["n_total"]
    x = np.linspace(n_total / pv.shape[1], n_total, pv.shape[1])
    crossed = (pv < 0.05).any(1)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for row, hit in zip(pv, crossed, strict=True):
        ax.plot(x, row, lw=1, color=COLORS[1] if hit else COLORS[0], alpha=0.85 if hit else 0.45)
    ax.axhline(0.05, color=INK, lw=1, ls="--")
    ax.text(x[0], 0.04, "p = 0.05", va="top", color=INK)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1)
    ax.set_xlabel("Набрано пользователей в группе")
    ax.set_ylabel("p-value (лог. шкала)")
    legend = [
        Line2D([], [], color=COLORS[1], label="хоть раз было p < 0.05"),
        Line2D([], [], color=COLORS[0], alpha=0.6, label="ни разу"),
    ]
    ax.legend(handles=legend, loc="lower right")
    ax.set_title(
        f"{crossed.sum()} из {len(pv)} A/A-тестов хотя бы раз опускались ниже 0.05 "
        f"за {pv.shape[1]} проверок"
    )
    return _save(fig, "pvalue_trajectories")


def power_curves(con):
    p = load_config()["scenarios"]["power"]["p"]
    df = _rates(con).query("scenario == 'power'").sort_values("n_per_group")
    n_grid = np.geomspace(df["n_per_group"].min(), df["n_per_group"].max(), 200)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for color, (u, d) in zip(COLORS, df.groupby("uplift"), strict=False):
        ax.plot(n_grid, power_two_proportions(p, u, n_grid), ls="--", lw=1.2, color=color)
        label = f"uplift +{u:.0%}"
        _points(
            ax, d["n_per_group"], d["rate"], d["ci_low"], d["ci_high"], color, label, line=False
        )
    ax.axhline(0.8, color=MUTED, lw=1, ls=":")
    ax.text(n_grid[0], 0.81, "80%", color=MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("Пользователей в группе (лог. шкала)")
    ax.set_ylabel("Мощность z-теста")
    _pct(ax)
    ax.legend(
        title="точки - симуляция,\nпунктир - формула",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        fontsize=9,
    )
    ax.set_title("Симуляция совпадает с формулой; вдвое меньший эффект требует ~4x выборку")
    return _save(fig, "power_curves")


def _by_test(ax, df, x, tests, offset):
    """Точки нескольких тестов рядом, со сдвигом по x, чтобы усы не накладывались."""
    for i, (color, test) in enumerate(zip(COLORS, tests, strict=False)):
        d = df[df["test"] == test].sort_values(x)
        shift = (i - (len(tests) - 1) / 2) * offset
        xs = d[x] * 10**shift if ax.get_xscale() == "log" else d[x] + shift
        _points(ax, xs, d["rate"], d["ci_low"], d["ci_high"], color, TEST_NAMES[test], line=False)


def fpr_by_test(con):
    df = _rates(con)
    aa = df.query("scenario == 'aa'")
    ratio = df.query("scenario == 'ratio' and uplift == 0").copy()
    levels = {"low": 0, "medium": 1, "high": 2}
    ratio["level"] = ratio["param"].str.extract(r"heterogeneity=(\w+)")[0].map(levels)

    panels = {
        "conversion": ["z_test", "welch_t", "bootstrap"],
        "revenue": ["welch_t", "mann_whitney", "bootstrap"],
        "ctr": ["naive_t", "delta", "bootstrap"],
        "pre_period": ["welch_t", "cuped", "bootstrap"],
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for ax, (metric, tests) in zip(axes.flat, panels.items(), strict=False):
        ax.set_xscale("log")
        _alpha_line(ax)
        _by_test(ax, aa[aa["metric"] == metric], "n_per_group", tests, 0.06)
        ax.set_title(METRIC_NAMES[metric], fontsize=11)
        ax.set_xlabel("Пользователей в группе")
        ax.set_ylim(0, 0.11)
        if metric == "ctr":
            ax.legend(
                fontsize=8,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0,
            )
        else:
            ax.legend(fontsize=8, loc="upper right")

    ax = axes.flat[4]
    _alpha_line(ax)
    _by_test(ax, ratio, "level", ["naive_t", "delta", "bootstrap"], 0.15)
    ax.set_xticks(range(3), labels=["низкая", "средняя", "высокая"])
    ax.set_xlabel("Разнородность пользователей по CTR")
    ax.set_title("CTR: наивный тест ломается с ростом разнородности", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")

    for ax in axes.flat[:5]:
        ax.set_ylabel("FPR (A/A)")
        _pct(ax)

    note = (
        "Серая полоса - 5% +- 0.43 п.п.:\nсюда должен попадать корректный\nтест при R = 10 000.\n\n"
        "Усы - 95% интервал Уилсона.\nДля бутстрапа R = 2 000,\nпоэтому усы шире."
    )
    axes.flat[5].axis("off")
    axes.flat[5].text(0, 0.5, note, color=MUTED, va="center")
    worst = ratio.query("test == 'naive_t'")["rate"].max()
    _suptitle(
        fig, f"Корректные тесты держат 5%, наивный тест CTR даёт до {worst:.0%} ложных срабатываний"
    )
    fig.tight_layout()
    return _save(fig, "fpr_by_test")


def pvalue_histograms(con, n=1000):
    df = db.query("simulations/07_pvalue_histogram", con).query("n_per_group == @n")
    combos = df[["metric", "test"]].drop_duplicates().values.tolist()
    rows = -(-len(combos) // 4)
    fig, axes = plt.subplots(rows, 4, figsize=(13, 2.6 * rows), sharey=True)
    for ax, (metric, test) in zip(axes.flat, combos, strict=False):
        h = df[(df["metric"] == metric) & (df["test"] == test)]
        ax.bar(
            h["bin_start"], h["share"], width=0.05, align="edge", color=COLORS[0], edgecolor="white"
        )
        ax.axhline(0.05, color=INK, lw=1, ls="--")
        ax.set_title(f"{METRIC_NAMES[metric]}\n{TEST_NAMES[test]}", fontsize=9)
        ax.set_xlim(0, 1)
    for ax in axes.flat[len(combos) :]:
        ax.axis("off")
    title = rf"При верной $H_0$ p-value распределены равномерно (n = {n}); пунктир - ожидаемые 5% в корзине"
    _suptitle(fig, title)
    fig.tight_layout()
    return _save(fig, "pvalue_histograms")


def heavy_tails_power(con):
    df = _rates(con).query("scenario == 'heavy_tails' and metric == 'revenue'").copy()
    df["sigma"] = df["param"].str.extract(r"sigma=([\d.]+)")[0].astype(float)
    df = df.sort_values("sigma")
    tests = ["welch_t", "bootstrap", "mann_whitney", "log_t", "winsorized_t"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for color, test in zip(COLORS, tests, strict=False):
        for ax, d in ((ax1, df[df["uplift"] > 0]), (ax2, df[df["uplift"] == 0])):
            d = d[d["test"] == test]
            _points(ax, d["sigma"], d["rate"], d["ci_low"], d["ci_high"], color, TEST_NAMES[test])
    _alpha_line(ax2)
    ax1.set_title(r"Мощность падает с ростом $\sigma$ чека", fontsize=11)
    ax1.set_ylabel(f"Мощность при uplift +{df['uplift'].max():.0%}")
    ax2.set_title(r"FPR в A/A при тех же $\sigma$", fontsize=11)
    ax2.set_ylabel("FPR (A/A)")
    for ax in (ax1, ax2):
        ax.set_xlabel(r"$\sigma$ логнормального чека")
        _pct(ax)
    ax1.legend(fontsize=8)
    _suptitle(
        fig,
        "Отсечение выбросов помогает только на тяжёлых хвостах, а log и Манн-Уитни почти не видят рост чека",
    )
    fig.tight_layout()
    return _save(fig, "heavy_tails_power")


def cuped_savings(con):
    df = db.query("simulations/05_cuped_variance", con)
    d = df[df["uplift"] > 0].sort_values("rho")
    rho = np.linspace(0, 0.95, 100)
    r7 = d.loc[np.isclose(d["rho"], 0.7), "var_ratio"].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax1.plot(
        rho, cuped_variance_ratio(rho), color=MUTED, ls="--", lw=1.2, label=r"теория $1 - \rho^2$"
    )
    lo, hi = d["var_ratio_low"], d["var_ratio_high"]
    _points(ax1, d["rho"], d["var_ratio"], lo, hi, COLORS[0], "симуляция", line=False)
    ax1.set_ylabel("Дисперсия оценки и нужный n\n(доля от обычного t-теста)")
    ax1.set_title(rf"При $\rho$ = 0.7 нужна {r7:.0%} выборки", fontsize=11)

    for color, col, name in (
        (COLORS[1], "rate_plain", "t-тест"),
        (COLORS[0], "rate_cuped", "CUPED"),
    ):
        lo, hi = wilson_ci(d[col] * d["reps"], d["reps"])
        _points(ax2, d["rho"], d[col], lo, hi, color, name)
    ax2.set_ylabel("Мощность")
    ax2.set_title("Мощность при том же n", fontsize=11)

    for ax in (ax1, ax2):
        ax.set_xlabel(r"Корреляция $\rho$ метрики с предпериодом")
        _pct(ax)
        ax.legend()
    title = rf"CUPED сокращает нужную выборку на {1 - r7:.0%} при $\rho$ = 0.7, как и предсказывает $1 - \rho^2$"
    _suptitle(fig, title)
    fig.tight_layout()
    return _save(fig, "cuped_savings")


def multiple_testing(con):
    df = db.query("simulations/04_fwer_fdr", con).query("n_true == 0").sort_values("k")
    k = np.arange(1, 21)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, rho in zip(axes, [0.0, 0.5], strict=True):
        for color, test in zip(COLORS, ["none", "bonferroni", "holm", "bh"], strict=False):
            d = df[(df["rho"] == rho) & (df["test"] == test)]
            _points(ax, d["k"], d["fwer"], d["fwer_low"], d["fwer_high"], color, TEST_NAMES[test])
        ax.plot(k, 1 - 0.95**k, color=MUTED, ls=":", lw=1.2, label=r"$1 - 0.95^k$")
        _alpha_line(ax, band=False)
        ax.set_xticks([1, 5, 10, 20])
        ax.set_xlabel("Число метрик k (все без эффекта)")
        title = "Независимые метрики" if rho == 0 else rf"Коррелированные метрики, $\rho$ = {rho}"
        ax.set_title(title, fontsize=11)
        _pct(ax)
    axes[0].set_ylabel("FWER: хоть одно ложное срабатывание")
    axes[0].legend(fontsize=8)
    f10 = df.query("rho == 0 and k == 10 and test == 'none'")["fwer"].iloc[0]
    _suptitle(
        fig,
        f"Без поправок 10 метрик дают {f10:.0%} шанс ложной находки; Бонферрони и Холм держат 5%",
    )
    fig.tight_layout()
    return _save(fig, "multiple_testing")


def make_all():
    setup_style()
    con = db.connect()
    return [
        peeking_fpr(con),
        pvalue_trajectories(),
        power_curves(con),
        fpr_by_test(con),
        pvalue_histograms(con),
        heavy_tails_power(con),
        cuped_savings(con),
        multiple_testing(con),
    ]
