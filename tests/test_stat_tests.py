"""Сверка наших критериев с SciPy и statsmodels."""

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

from abtest_sim import generators as gen
from abtest_sim import stat_tests as st

TOL = 1e-8


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_z_test_matches_statsmodels(rng):
    a, b = gen.conversion(3000, 0.1, 0.1, rng)
    ours = st.z_test_proportions(a, b)
    _, p_ref = proportions_ztest([b.sum(), a.sum()], [b.size, a.size])
    assert ours.p_value == pytest.approx(p_ref, abs=TOL)


def test_z_test_vectorized_equals_rowwise(rng):
    a, b = gen.conversion(500, 0.1, 0.0, rng, reps=20)
    batch = st.z_test_proportions(a, b)
    for i in range(20):
        _, p_ref = proportions_ztest([b[i].sum(), a[i].sum()], [500, 500])
        assert batch.p_value[i] == pytest.approx(p_ref, abs=TOL)


def test_welch_matches_scipy(rng):
    a, b = gen.revenue(800, 0.1, 6, 1.2, 0.2, rng, reps=5)
    ours = st.welch_t(a, b)
    ref = stats.ttest_ind(b, a, axis=-1, equal_var=False)
    np.testing.assert_allclose(ours.p_value, ref.pvalue, atol=TOL)
    ci = ref.confidence_interval()
    np.testing.assert_allclose(ours.ci_low, ci.low, rtol=1e-8)
    np.testing.assert_allclose(ours.ci_high, ci.high, rtol=1e-8)


@pytest.mark.parametrize("kind", ["continuous", "ties"])
def test_mann_whitney_matches_scipy(rng, kind):
    if kind == "continuous":
        a = rng.normal(0, 1, size=(4, 300))
        b = rng.normal(0.1, 1, size=(4, 250))
    else:  # выручка: 90% нулей, много одинаковых значений
        a, b = gen.revenue(400, 0.1, 6, 1.2, 0.0, rng, reps=4)
    ours = st.mann_whitney(a, b)
    ref = stats.mannwhitneyu(b, a, axis=-1, method="asymptotic", use_continuity=True)
    np.testing.assert_allclose(ours.p_value, ref.pvalue, atol=TOL)
    np.testing.assert_allclose(ours.effect, ref.statistic / (a.shape[1] * b.shape[1]), atol=TOL)


def test_mann_whitney_1d(rng):
    a, b = rng.normal(size=100), rng.normal(size=120)
    ours = st.mann_whitney(a, b)
    ref = stats.mannwhitneyu(b, a, method="asymptotic")
    assert np.ndim(ours.p_value) == 0
    assert ours.p_value == pytest.approx(ref.pvalue, abs=TOL)


def test_bootstrap_ci_close_to_scipy(rng):
    a = rng.lognormal(0, 1, 2000)
    b = rng.lognormal(0.05, 1, 2000)
    ours = st.bootstrap_mean_diff(a, b, rng, n_boot=4000)
    ref = stats.bootstrap(
        (b, a),
        lambda x, y, axis: x.mean(axis) - y.mean(axis),
        n_resamples=4000,
        method="percentile",
        rng=np.random.default_rng(1),
    )
    # перевыборки случайные, поэтому сверяем с запасом на погрешность
    se = a.std() * np.sqrt(2 / 2000)
    assert ours.ci_low == pytest.approx(ref.confidence_interval.low, abs=0.1 * se)
    assert ours.ci_high == pytest.approx(ref.confidence_interval.high, abs=0.1 * se)


def test_bootstrap_p_value_consistent_with_ci(rng):
    a, b = rng.normal(0, 1, (30, 300)), rng.normal(0.15, 1, (30, 300))
    res = st.bootstrap_mean_diff(a, b, rng, n_boot=500)
    outside = (res.ci_low > 0) | (res.ci_high < 0)
    assert np.mean(outside == (res.p_value < 0.05)) >= 0.95


def test_delta_method_matches_user_bootstrap(rng):
    s = gen.ctr(3000, 10, 2, 2, 38, 0.0, rng)
    delta = st.delta_ratio(*s)
    boot = st.bootstrap_ratio(*s, rng=rng, n_boot=4000)
    width_delta = delta.ci_high - delta.ci_low
    width_boot = boot.ci_high - boot.ci_low
    assert width_delta == pytest.approx(width_boot, rel=0.07)
    assert delta.effect == pytest.approx(boot.effect, abs=TOL)


def test_naive_ratio_underestimates_variance(rng):
    s = gen.ctr(3000, 10, 2, 0.2, 3.8, 0.0, rng)  # очень разнородные пользователи
    naive = st.naive_ratio_t(*s)
    delta = st.delta_ratio(*s)
    assert naive.ci_high - naive.ci_low < delta.ci_high - delta.ci_low


def test_cuped_reduces_variance_and_keeps_effect(rng):
    s = gen.pre_period(2000, 100, 20, 0.7, 0.0, rng, reps=200)
    plain = st.welch_t(s.y_a, s.y_b)
    adj = st.cuped(s.y_a, s.x_a, s.y_b, s.x_b)
    ratio = adj.effect.var() / plain.effect.var()
    assert ratio == pytest.approx(1 - 0.7**2, abs=0.1)
    assert abs(adj.effect.mean()) < 3 * adj.effect.std() / np.sqrt(200)


def test_cuped_theta_is_regression_slope(rng):
    x = rng.normal(size=1000)
    y = 2.5 * x + rng.normal(size=1000)
    assert st.cuped_theta(x, y) == pytest.approx(stats.linregress(x, y).slope, abs=TOL)


def test_srm_matches_scipy():
    ours = st.srm_chi2(np.array([50_500, 49_000]), np.array([49_500, 51_000]))
    for i, (na, nb) in enumerate([(50_500, 49_500), (49_000, 51_000)]):
        ref = stats.chisquare([na, nb], [(na + nb) / 2] * 2)
        assert ours.p_value[i] == pytest.approx(ref.pvalue, abs=TOL)


@pytest.mark.parametrize(("total", "chi2"), [(10_000, 1), (100_000, 10), (1_000_000, 100)])
def test_srm_control_points(total, chi2):
    # контрольные точки сценария 8: разбиение 50.5 / 49.5
    res = st.srm_chi2(total * 0.495, total * 0.505)
    assert res.p_value == pytest.approx(stats.chi2.sf(chi2, df=1), rel=1e-6)


def test_wilson_ci_matches_statsmodels():
    from statsmodels.stats.proportion import proportion_confint

    lo, hi = st.wilson_ci(500, 10_000)
    ref_lo, ref_hi = proportion_confint(500, 10_000, method="wilson")
    assert lo == pytest.approx(ref_lo, abs=TOL)
    assert hi == pytest.approx(ref_hi, abs=TOL)
    assert (hi - lo) / 2 == pytest.approx(0.0043, abs=0.0001)


def test_winsorized_and_log_run(rng):
    a, b = gen.revenue(500, 0.1, 6, 2.0, 0.0, rng, reps=3)
    for fn in (st.log_t, st.winsorized_t):
        res = fn(a, b)
        assert res.p_value.shape == (3,)
        assert np.all((res.p_value >= 0) & (res.p_value <= 1))
