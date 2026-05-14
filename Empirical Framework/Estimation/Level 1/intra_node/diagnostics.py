"""
Intensity tests, consistency function, and dissonance metric for L1 intra-node.

This module operationalises the three measurement objects from the
Level 1 Estimation document, applied to *coefficient drift* (intra-L1):

  1. Intensity test:
       - R²(static fit on calibration window) > 0.7
       - σ(ε)/σ(i) < 0.4
       - Granger causality from {regressors} → i_t, p < 0.05
       - Out-of-sample RMSE on the held-out test window (diagnostic, not gating)

  2. Consistency function (intra-L1):

       R^{L1_intra}(θ̂_t; θ̂_baseline) = θ̂_t − θ̂_baseline           (vector)

     By construction E[R] = 0 over the calibration window where θ̂_baseline is
     defined as the static mean.

  3. Dissonance function (intra-L1):

       D^{L1_intra}_t = ‖θ̂_t − θ̂_baseline‖_Σ                         (scalar)

     The default norm is the Mahalanobis distance with Σ = cov(θ̂_baseline) — this
     makes the metric scale-invariant across coefficients. An unweighted L2 norm
     and per-coefficient absolute drift are also computed for diagnostic use.

  4. Threshold (intra-L1):

       τ = 2 · σ(D) on the calibration window  (pre-registered headline)

     Plus robustness across {1.5σ, 2σ, 2.5σ} and percentiles {90, 95, 99}.

Three auxiliary diagnostics are carried separately, never folded into D:
  - sign of drift per coefficient (hawkish/dovish bias evolving)
  - persistence (AR(1) of |drift|)
  - cumulative drift over a 12-month rolling window
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Mapping

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from cb_dissonance.src.data.schema import (
    ReactionFunctionSpec,
    COL_POLICY_RATE,
)
from cb_dissonance.src.level_1.intra_node.reaction_function import (
    StaticFitResult,
    fit_static,
)
from cb_dissonance.src.level_1.intra_node.tvp_estimation import TVPFitResult
from cb_dissonance.src.level_1.intra_node.rolling_estimation import RollingFitResult


# =============================================================================
# 1. Intensity tests
# =============================================================================

@dataclass
class IntensityReport:
    country: str
    r_squared: float
    residual_std_ratio: float
    out_of_sample_rmse: Optional[float]
    granger_p_values: dict[str, float]
    passes_r_squared: bool
    passes_residual_ratio: bool
    passes_granger: bool

    @property
    def passes_sufficiency(self) -> bool:
        return self.passes_r_squared and self.passes_residual_ratio and self.passes_granger

    def to_series(self) -> pd.Series:
        flat = {
            "country": self.country,
            "r_squared": self.r_squared,
            "residual_std_ratio": self.residual_std_ratio,
            "out_of_sample_rmse": self.out_of_sample_rmse,
            "passes_sufficiency": self.passes_sufficiency,
        }
        for k, v in self.granger_p_values.items():
            flat[f"granger_p_{k}"] = v
        return pd.Series(flat)


def intensity_test(
    df: pd.DataFrame,
    spec: ReactionFunctionSpec,
    static_fit: StaticFitResult,
    test_window_data: Optional[pd.DataFrame] = None,
    r_squared_min: float = 0.7,
    residual_ratio_max: float = 0.4,
    granger_p_max: float = 0.05,
    granger_maxlag: int = 4,
) -> IntensityReport:
    """
    Run all four intensity diagnostics. Returns a structured report.

    Parameters
    ----------
    df : full sample data (loader output).
    spec : country spec.
    static_fit : output of ``fit_static`` over the calibration window.
    test_window_data : if provided, used to compute out-of-sample RMSE.
        Typically you slice df to spec.test_window and pass it here. If None,
        out-of-sample RMSE is not computed.
    """
    # R² and residual ratio from the static fit
    r2 = static_fit.r_squared
    sigma_eps = float(static_fit.residuals.std())
    sigma_i = float(
        df.set_index("date").loc[
            slice(*spec.calibration_window), COL_POLICY_RATE
        ].std()
    )
    ratio = sigma_eps / sigma_i if sigma_i > 0 else np.inf

    # Out-of-sample RMSE
    oos_rmse: Optional[float] = None
    if test_window_data is not None:
        from cb_dissonance.src.data.schema import build_regressor_matrix, COL_DATE
        X_test = build_regressor_matrix(test_window_data, spec)
        X_test.index = pd.to_datetime(X_test.index)
        X_test = X_test.dropna()
        y_test = (
            test_window_data.set_index(pd.to_datetime(test_window_data[COL_DATE]))
            [COL_POLICY_RATE]
            .loc[X_test.index]
        )
        i_star = static_fit.predict(X_test)
        oos_rmse = float(np.sqrt(((y_test - i_star) ** 2).mean()))

    # Granger causality — each regressor individually, controlling on its own lag
    granger_ps: dict[str, float] = {}
    cal_window = df[
        (df["date"] >= spec.calibration_window[0])
        & (df["date"] <= spec.calibration_window[1])
    ].copy()

    # Build the same regressor matrix used in estimation so the test is consistent
    from cb_dissonance.src.data.schema import build_regressor_matrix
    X_cal = build_regressor_matrix(cal_window, spec)
    X_cal.index = pd.to_datetime(X_cal.index)
    y_cal = (
        cal_window.set_index(pd.to_datetime(cal_window["date"]))
        [COL_POLICY_RATE]
    )
    joined = pd.concat([y_cal.rename("y"), X_cal], axis=1).dropna()

    for col in X_cal.columns:
        if col == "i_lag1":
            # Skip — Granger of i_lag1 on i is trivially significant
            continue
        try:
            test_data = joined[["y", col]].dropna()
            if len(test_data) < granger_maxlag * 4:
                granger_ps[col] = np.nan
                continue
            gres = grangercausalitytests(test_data, maxlag=granger_maxlag, verbose=False)
            # Take the minimum p-value across lags (most permissive test of any causality)
            min_p = min(gres[lag][0]["ssr_ftest"][1] for lag in gres)
            granger_ps[col] = float(min_p)
        except Exception:
            granger_ps[col] = np.nan

    passes_r2 = r2 >= r_squared_min
    passes_ratio = ratio <= residual_ratio_max
    # Granger: at least one regressor must Granger-cause i_t
    passes_granger = any(
        (p < granger_p_max) for p in granger_ps.values() if not np.isnan(p)
    )

    return IntensityReport(
        country=spec.country,
        r_squared=float(r2),
        residual_std_ratio=float(ratio),
        out_of_sample_rmse=oos_rmse,
        granger_p_values=granger_ps,
        passes_r_squared=passes_r2,
        passes_residual_ratio=passes_ratio,
        passes_granger=passes_granger,
    )


# =============================================================================
# 2. Consistency function (intra-L1)
# =============================================================================

def consistency_intra_l1(
    coefficients_over_time: pd.DataFrame,
    baseline: pd.Series,
    coefficient_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    R^{L1_intra}_t = θ̂_t − θ̂_baseline   (per-coefficient signed drift)

    Parameters
    ----------
    coefficients_over_time : DataFrame, rows = dates, cols = coefficient names.
        Pass either filtered/smoothed TVP coefficients or rolling-window
        coefficients. Use structural (α, β, …) — not reduced form — so the
        baseline anchor is interpretable.
    baseline : Series of baseline values keyed by the same coefficient names.
        Typically static_fit.structural_coefficients.
    coefficient_columns : optional subset of coefficients to compute R for.
        Defaults to the intersection of the two index sets.

    Returns
    -------
    DataFrame of signed drifts. E[R] = 0 in the calibration window by
    construction (if baseline = mean of θ̂_t over that window).
    """
    if coefficient_columns is None:
        coefficient_columns = sorted(
            set(coefficients_over_time.columns) & set(baseline.index)
        )
    aligned = coefficients_over_time[coefficient_columns]
    base = baseline[coefficient_columns]
    return aligned.subtract(base, axis=1)


# =============================================================================
# 3. Dissonance function (intra-L1)
# =============================================================================

@dataclass
class DissonanceTrace:
    """Full per-time-point dissonance output, including auxiliary dimensions."""

    spec: ReactionFunctionSpec
    consistency: pd.DataFrame          # R_t per coefficient
    d_mahalanobis: pd.Series           # headline scalar
    d_l2: pd.Series                    # unweighted L2 norm comparator
    d_per_coef_abs: pd.DataFrame       # |R_t| per coefficient — for attribution
    sign_per_coef: pd.DataFrame        # +1 / -1 / 0 — drift direction
    persistence_per_coef: pd.Series    # AR(1) of |R_t,j| on 12-period window
    cumulative_drift: pd.DataFrame     # Σ_{t-11:t} R_t per coefficient
    threshold: float                   # τ headline
    threshold_robustness: dict[str, float]  # τ at alternative k_sigma & percentiles
    fires: pd.Series                   # bool — d_mahalanobis > threshold
    method_used: str                   # "TVP_filtered" | "rolling"


def dissonance_intra_l1(
    coefficients_over_time: pd.DataFrame,
    baseline: pd.Series,
    spec: ReactionFunctionSpec,
    calibration_cov: Optional[np.ndarray] = None,
    k_sigma: float = 2.0,
    robustness_k_sigmas: tuple[float, ...] = (1.5, 2.0, 2.5),
    robustness_percentiles: tuple[float, ...] = (0.90, 0.95, 0.99),
    method_label: str = "TVP_filtered",
    burn_in_periods: int = 24,
) -> DissonanceTrace:
    """
    Compute D^{L1_intra}_t and its auxiliary decomposition.

    Parameters
    ----------
    coefficients_over_time : θ̂_t (structural, e.g. TVPFitResult.filtered_structural).
    baseline : θ̂_baseline (structural, from StaticFitResult.structural_coefficients).
    spec : country spec.
    calibration_cov : optional covariance matrix of θ̂ over the calibration window.
        Used as Σ in the Mahalanobis norm. If None, computed as
        cov(θ̂_t) over spec.calibration_window. Setting this is recommended for
        TVP fits where the within-window dispersion is informative.
    k_sigma : multiplier for the threshold τ = k · σ(D).
    robustness_k_sigmas, robustness_percentiles : alternative thresholds.
    burn_in_periods : drop the first N observations from BOTH calibration cov
        estimation AND from the firing series. Kalman filters need 12-24 periods
        to converge from a diffuse prior; treating those transients as drift
        produces systematic false positives at the start of the sample. The
        default of 24 monthly periods (2 years) is conservative.

    Returns
    -------
    DissonanceTrace
    """
    # Restrict to coefficients that appear in both
    coef_cols = sorted(set(coefficients_over_time.columns) & set(baseline.index))
    theta = coefficients_over_time[coef_cols].copy()
    base = baseline[coef_cols]

    # Apply burn-in: NaN-out the first burn_in_periods observations
    if burn_in_periods > 0 and len(theta) > burn_in_periods:
        burn_in_index = theta.index[:burn_in_periods]
        theta.loc[burn_in_index] = np.nan

    R = consistency_intra_l1(theta, base, coef_cols)

    # Covariance estimate from the calibration window (post burn-in)
    cal_start, cal_end = (pd.Timestamp(d) for d in spec.calibration_window)
    cal_mask = (R.index >= cal_start) & (R.index <= cal_end)
    R_cal = R.loc[cal_mask].dropna()

    if calibration_cov is None:
        if len(R_cal) < 2:
            raise ValueError(
                f"{spec.country}: insufficient observations in calibration window "
                f"to estimate covariance ({len(R_cal)} rows). "
                f"Consider reducing burn_in_periods or widening calibration_window."
            )
        Sigma = np.cov(R_cal.values.T)
        # Regularise if near-singular
        Sigma = Sigma + 1e-8 * np.eye(Sigma.shape[0])
    else:
        Sigma = calibration_cov

    Sigma_inv = np.linalg.pinv(Sigma)

    # Compute on non-NaN rows; rows in burn-in get NaN
    R_filled = R.values
    valid_mask = ~np.isnan(R_filled).any(axis=1)
    d_mahal_arr = np.full(len(R), np.nan)
    d_l2_arr = np.full(len(R), np.nan)

    R_valid = R_filled[valid_mask]
    d_mahal_arr[valid_mask] = np.sqrt(
        np.einsum("ti,ij,tj->t", R_valid, Sigma_inv, R_valid)
    )
    d_l2_arr[valid_mask] = np.linalg.norm(R_valid, axis=1)

    d_mahal_s = pd.Series(d_mahal_arr, index=R.index, name="d_mahalanobis")
    d_l2_s = pd.Series(d_l2_arr, index=R.index, name="d_l2")

    # Per-coefficient absolute drift
    d_abs = R.abs()

    # Sign per coefficient (Int64 to handle NaNs from burn-in)
    sign = np.sign(R).astype("Int64")

    # Persistence: AR(1) of |R_t,j| on the full sample (post burn-in)
    def _ar1(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if len(x) < 3:
            return np.nan
        x1, x0 = x[1:], x[:-1]
        if np.std(x0) == 0:
            return np.nan
        return float(np.corrcoef(x0, x1)[0, 1])

    persistence = pd.Series(
        {c: _ar1(d_abs[c].values) for c in d_abs.columns},
        name="ar1_abs_drift",
    )

    cumulative = R.rolling(12, min_periods=6).sum()

    # Threshold from D's distribution on the calibration window
    d_cal = d_mahal_s.loc[cal_mask].dropna()
    sigma_D = float(d_cal.std()) if len(d_cal) >= 2 else np.nan
    tau = k_sigma * sigma_D if not np.isnan(sigma_D) else np.nan

    robustness = {}
    for k in robustness_k_sigmas:
        robustness[f"{k:.1f}_sigma"] = float(k * sigma_D) if not np.isnan(sigma_D) else np.nan
    for p in robustness_percentiles:
        robustness[f"p{int(p*100)}"] = float(d_cal.quantile(p)) if len(d_cal) > 0 else np.nan

    fires = (d_mahal_s > tau).fillna(False).rename("fires")

    return DissonanceTrace(
        spec=spec,
        consistency=R,
        d_mahalanobis=d_mahal_s,
        d_l2=d_l2_s,
        d_per_coef_abs=d_abs,
        sign_per_coef=sign,
        persistence_per_coef=persistence,
        cumulative_drift=cumulative,
        threshold=float(tau) if not np.isnan(tau) else float("nan"),
        threshold_robustness=robustness,
        fires=fires,
        method_used=method_label,
    )


# =============================================================================
# Convenience: end-to-end pipeline
# =============================================================================

def run_l1_intra_pipeline(
    df: pd.DataFrame,
    spec: ReactionFunctionSpec,
    test_window_data: Optional[pd.DataFrame] = None,
    tvp_initial_state_from_static: bool = True,
) -> dict:
    """
    Convenience wrapper: static → intensity → TVP → dissonance, with TVP
    optionally initialised from the static fit. Returns a dict of all artefacts
    so downstream code (notebook, composite aggregation) has a single handle.

    Notes
    -----
    Rolling-window fit is *not* run here by default — call ``fit_rolling``
    separately when you want the comparator. Reason: rolling is O(T·window) and
    typically only run once per pre-registration cycle.
    """
    from cb_dissonance.src.level_1.intra_node.tvp_estimation import fit_tvp

    static = fit_static(df, spec, method="OLS")
    intensity = intensity_test(df, spec, static, test_window_data=test_window_data)

    initial_state = None
    if tvp_initial_state_from_static:
        # Order must match the TVP model's exog matrix: const, then regressor cols
        # in build_regressor_matrix order. statsmodels.add_constant prepends 'const'.
        cols = ["const"] + spec.regressors
        initial_state = static.reduced_form_coefficients.reindex(cols).fillna(0).values

    tvp = fit_tvp(df, spec, initial_state=initial_state)

    diss = dissonance_intra_l1(
        coefficients_over_time=tvp.filtered_structural,
        baseline=static.structural_coefficients,
        spec=spec,
        method_label="TVP_filtered",
    )

    return {
        "static_fit": static,
        "intensity": intensity,
        "tvp_fit": tvp,
        "dissonance": diss,
    }
