"""
Unit tests for L1 intra-node: synthetic ground-truth recovery.

Strategy: generate clean synthetic data with known coefficients and known
regime shifts, then verify the estimators recover what we put in. These are
ground-truth tests, not regression tests — if a refactor breaks them, the
econometrics has actually broken.

Run with: pytest cb_dissonance/tests/test_l1_intra.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cb_dissonance.src.data.schema import (
    ReactionFunctionSpec,
    COL_DATE,
    COL_POLICY_RATE,
    COL_INFLATION,
    COL_INFLATION_TARGET,
    COL_OUTPUT_GAP,
    COL_INFLATION_EXP_H,
    COL_REER_CHANGE,
    COL_FOREIGN_RATE,
)
from cb_dissonance.src.level_1.intra_node.reaction_function import (
    fit_static,
    passes_taylor_principle,
)
from cb_dissonance.src.level_1.intra_node.rolling_estimation import fit_rolling
from cb_dissonance.src.level_1.intra_node.tvp_estimation import fit_tvp
from cb_dissonance.src.level_1.intra_node.diagnostics import (
    intensity_test,
    consistency_intra_l1,
    dissonance_intra_l1,
)


# =============================================================================
# Synthetic data generators
# =============================================================================

def _make_spec(include_fx: bool = True, include_foreign: bool = True) -> ReactionFunctionSpec:
    return ReactionFunctionSpec(
        country="TST",
        spec_type="AM_MK_EM",
        pi_star=4.5,
        r_star=2.0,
        inflation_horizon=4,
        output_gap_horizon=0,
        smoothing=True,
        include_fx=include_fx,
        include_foreign_rate=include_foreign,
        calibration_window=("2005-01-01", "2014-12-31"),
        test_window=("2015-01-01", "2024-12-31"),
        frequency="M",
        panel="EM",
    )


def _make_stable_panel(
    true_alpha: float = 1.5,
    true_beta: float = 0.5,
    true_rho: float = 0.8,
    true_gamma: float = -0.05,
    true_delta: float = 0.3,
    sigma_eps: float = 0.10,
    T: int = 240,
    seed: int = 7,
) -> pd.DataFrame:
    """Clean DGP with constant coefficients (no drift)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-31", periods=T, freq="ME")
    pi_star, r_star = 4.5, 2.0

    inflation = np.clip(4.5 + np.cumsum(rng.normal(0, 0.15, T)), 1.0, 12.0)
    output_gap = np.clip(np.cumsum(rng.normal(0, 0.10, T)), -4.0, 4.0)
    # REER change variance bumped up so γ is identifiable with σ_ε=0.10
    reer_change = rng.normal(0, 0.10, T)
    us_ffr = np.clip(2.0 + np.cumsum(rng.normal(0, 0.10, T)), 0.0, 6.0)
    infl_exp = pd.Series(inflation).shift(-4).ffill().values + rng.normal(0, 0.2, T)

    i = np.zeros(T)
    i[0] = r_star + pi_star
    for t in range(1, T):
        x_pi = true_alpha * (infl_exp[t] - pi_star)
        x_y = true_beta * output_gap[t]
        x_q = true_gamma * reer_change[t]
        x_f = true_delta * us_ffr[t]
        i_star = r_star + pi_star + x_pi + x_y + x_q + x_f
        i[t] = true_rho * i[t - 1] + (1 - true_rho) * i_star + rng.normal(0, sigma_eps)

    return pd.DataFrame({
        COL_DATE: dates,
        COL_POLICY_RATE: i,
        COL_INFLATION: inflation,
        COL_INFLATION_TARGET: pi_star,
        COL_OUTPUT_GAP: output_gap,
        COL_INFLATION_EXP_H: infl_exp,
        COL_REER_CHANGE: reer_change,
        COL_FOREIGN_RATE: us_ffr,
    })


def _make_drift_panel(
    alpha_pre: float = 1.5,
    alpha_post: float = 0.9,
    break_idx: int = 156,
    T: int = 240,
    seed: int = 11,
) -> pd.DataFrame:
    """Panel with an α regime shift at break_idx with a 12m ramp."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-31", periods=T, freq="ME")
    pi_star, r_star = 4.5, 2.0

    inflation = np.clip(4.5 + np.cumsum(rng.normal(0, 0.15, T)), 1.0, 12.0)
    output_gap = np.clip(np.cumsum(rng.normal(0, 0.10, T)), -4.0, 4.0)
    reer_change = rng.normal(0, 0.10, T)
    us_ffr = np.clip(2.0 + np.cumsum(rng.normal(0, 0.10, T)), 0.0, 6.0)
    infl_exp = pd.Series(inflation).shift(-4).ffill().values + rng.normal(0, 0.2, T)

    alpha_t = np.full(T, alpha_pre)
    for t in range(break_idx, T):
        k = t - break_idx
        alpha_t[t] = alpha_pre - (alpha_pre - alpha_post) * min(k / 12, 1.0)

    beta, rho, gamma, delta = 0.5, 0.8, -0.05, 0.3
    i = np.zeros(T)
    i[0] = r_star + pi_star
    for t in range(1, T):
        x_pi = alpha_t[t] * (infl_exp[t] - pi_star)
        x_y = beta * output_gap[t]
        x_q = gamma * reer_change[t]
        x_f = delta * us_ffr[t]
        i_star = r_star + pi_star + x_pi + x_y + x_q + x_f
        i[t] = rho * i[t - 1] + (1 - rho) * i_star + rng.normal(0, 0.10)

    return pd.DataFrame({
        COL_DATE: dates,
        COL_POLICY_RATE: i,
        COL_INFLATION: inflation,
        COL_INFLATION_TARGET: pi_star,
        COL_OUTPUT_GAP: output_gap,
        COL_INFLATION_EXP_H: infl_exp,
        COL_REER_CHANGE: reer_change,
        COL_FOREIGN_RATE: us_ffr,
    })


# =============================================================================
# Static recovery
# =============================================================================

class TestStaticRecovery:
    def test_recovers_known_coefficients(self):
        df = _make_stable_panel()
        spec = _make_spec()
        fit = fit_static(df, spec, method="OLS")
        s = fit.structural_coefficients
        assert abs(s["alpha"] - 1.5) < 0.30, f"α off: {s['alpha']}"
        assert abs(s["beta"] - 0.5) < 0.25, f"β off: {s['beta']}"
        assert abs(s["rho"] - 0.8) < 0.10, f"ρ off: {s['rho']}"
        assert abs(s["delta"] - 0.3) < 0.25, f"δ off: {s['delta']}"
        assert abs(s["gamma"] - (-0.05)) < 0.50

    def test_high_r_squared_on_clean_dgp(self):
        df = _make_stable_panel(sigma_eps=0.10)
        spec = _make_spec()
        fit = fit_static(df, spec, method="OLS")
        assert fit.r_squared > 0.95, f"R² = {fit.r_squared}"

    def test_taylor_principle_check(self):
        df_hawk = _make_stable_panel(true_alpha=1.5)
        df_dove = _make_stable_panel(true_alpha=0.9, seed=23)
        spec = _make_spec()
        assert passes_taylor_principle(fit_static(df_hawk, spec))
        assert not passes_taylor_principle(fit_static(df_dove, spec))


# =============================================================================
# Rolling detection
# =============================================================================

class TestRollingDetection:
    def test_detects_alpha_regime_shift(self):
        df = _make_drift_panel(alpha_pre=1.5, alpha_post=0.9)
        spec = _make_spec()
        roll = fit_rolling(df, spec, window=60)
        alpha_path = roll.structural_coefficients["alpha"].dropna()
        assert alpha_path.min() < 1.0, "rolling didn't detect dovish regime"
        assert alpha_path.max() > 1.3, "rolling didn't capture hawkish regime"

    def test_no_false_drift_on_stable_dgp(self):
        df = _make_stable_panel()
        spec = _make_spec()
        roll = fit_rolling(df, spec, window=60)
        alpha_path = roll.structural_coefficients["alpha"].dropna()
        # Use IQR rather than min-max: structural α = a/(1-ρ̂) can spike on short
        # windows where ρ̂ approaches 1 — this is a known limitation of recovering
        # α from a smoothed Taylor rule. The mass of the distribution is the
        # diagnostic; tails are noise.
        iqr = alpha_path.quantile(0.75) - alpha_path.quantile(0.25)
        assert iqr < 1.0, (
            f"rolling α IQR too wide on stable DGP: {iqr:.3f}"
        )
        # Median should stay close to truth (α=1.5)
        assert abs(alpha_path.median() - 1.5) < 0.5, (
            f"rolling α median {alpha_path.median():.3f} far from truth 1.5"
        )


# =============================================================================
# TVP detection (with fixed Q to bypass pile-up)
# =============================================================================

class TestTVPDetection:
    def test_tvp_with_fixed_q_detects_shift(self):
        df = _make_drift_panel(alpha_pre=1.5, alpha_post=0.9)
        spec = _make_spec()
        static = fit_static(df.iloc[:120], spec, method="OLS")
        p = len(static.reduced_form_coefficients)
        tvp = fit_tvp(
            df, spec,
            initial_state=static.reduced_form_coefficients.values,
            fixed_q_diag=np.full(p, 1e-3),
            fixed_h=0.01,
        )
        alpha_path = tvp.filtered_structural["alpha"]
        pre = alpha_path.iloc[24:120].mean()
        post = alpha_path.iloc[170:].mean()
        assert post < pre, f"TVP didn't detect α decline: pre={pre:.3f} post={post:.3f}"
        assert (pre - post) > 0.15, (
            f"TVP detected shift but too weakly: pre={pre:.3f} post={post:.3f}"
        )


# =============================================================================
# Intensity gating
# =============================================================================

class TestIntensity:
    def test_passes_on_clean_dgp(self):
        df = _make_stable_panel(sigma_eps=0.10)
        spec = _make_spec()
        static = fit_static(df, spec, method="OLS")
        report = intensity_test(df, spec, static)
        assert report.passes_r_squared
        assert report.passes_residual_ratio
        assert report.passes_granger
        assert report.passes_sufficiency

    def test_fails_on_noise(self):
        df = _make_stable_panel()
        rng = np.random.default_rng(99)
        df[COL_POLICY_RATE] = rng.normal(6.0, 1.0, len(df))
        spec = _make_spec()
        static = fit_static(df, spec, method="OLS")
        report = intensity_test(df, spec, static)
        assert not report.passes_sufficiency, "Intensity passed on pure noise — bug"


# =============================================================================
# Consistency and dissonance
# =============================================================================

class TestConsistencyDissonance:
    def test_consistency_near_zero_on_calibration(self):
        df = _make_stable_panel()
        spec = _make_spec()
        roll = fit_rolling(df, spec, window=60)
        cal = roll.structural_coefficients.loc[spec.calibration_window[0]:spec.calibration_window[1]]
        baseline = cal.mean()
        R = consistency_intra_l1(roll.structural_coefficients, baseline)
        cal_R = R.loc[spec.calibration_window[0]:spec.calibration_window[1]]
        # By construction E[R_t] over cal window = 0 because baseline is the cal-window mean.
        # Test the column means rather than the abs means (which always exceed 0).
        col_means = cal_R.mean().abs()
        assert (col_means < 1e-6).all(), col_means

    def test_threshold_robustness_monotone(self):
        df = _make_drift_panel()
        spec = _make_spec()
        roll = fit_rolling(df, spec, window=60)
        cal = roll.structural_coefficients.loc[spec.calibration_window[0]:spec.calibration_window[1]]
        baseline = cal.mean()
        trace = dissonance_intra_l1(roll.structural_coefficients, baseline, spec)
        thr = trace.threshold_robustness
        assert thr["1.5_sigma"] < thr["2.0_sigma"] < thr["2.5_sigma"]
        assert thr["p90"] < thr["p95"] < thr["p99"]

    def test_dissonance_fires_after_break(self):
        df = _make_drift_panel()
        spec = _make_spec()
        roll = fit_rolling(df, spec, window=60)
        cal = roll.structural_coefficients.loc[spec.calibration_window[0]:spec.calibration_window[1]]
        baseline = cal.mean()
        trace = dissonance_intra_l1(roll.structural_coefficients, baseline, spec)
        D = trace.d_mahalanobis.dropna()
        firing = (D > trace.threshold)
        pre = firing.loc[:spec.calibration_window[1]].mean()
        post = firing.loc[spec.test_window[0]:].mean()
        assert post > pre, (
            f"post-break firing ({post:.2f}) not greater than pre-break ({pre:.2f})"
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
