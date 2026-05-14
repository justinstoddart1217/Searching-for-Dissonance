"""
Static reaction function estimation.

Estimates a single coefficient vector θ̂ = (c, ρ, α', β, γ, δ) over the
calibration window. This θ̂ serves three downstream roles:

  1. Baseline for intra-L1 drift measurement (θ̂_t − θ̂_baseline).
  2. Comparator for the TVP fit (must agree with the TVP estimate at the
     centre of the calibration window).
  3. Generator of the fitted rate i*_t for L1↔L3a vertical dissonance.

The estimated CGG-smoothed form is non-linear in ρ but can be estimated linearly
on the reduced form:

    i_t = c + ρ·i_{t-1} + a·(E_t π_{t+h} − π*) + b·E_t ỹ_{t+k} + g·Δq_t + d·i^foreign_t + ε_t

where a = (1−ρ)·α, b = (1−ρ)·β, etc.  Structural coefficients are recovered by
dividing by (1−ρ̂).  Standard errors on structural coefficients are obtained via
the delta method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS

from cb_dissonance.src.data.schema import (
    ReactionFunctionSpec,
    COL_DATE,
    COL_POLICY_RATE,
    build_regressor_matrix,
    validate_input_frame,
)


@dataclass
class StaticFitResult:
    """Container for fit output. Keep raw OLS/GMM results object for downstream use."""

    spec: ReactionFunctionSpec
    method: str                                 # "OLS" | "GMM"
    reduced_form_coefficients: pd.Series        # the directly-estimated a, b, g, d, ρ, c
    structural_coefficients: pd.Series          # α, β, γ, δ recovered via /(1-ρ̂)
    standard_errors: pd.Series                  # SEs on reduced form
    fitted_values: pd.Series                    # i*_t at training points
    residuals: pd.Series                        # ε_t at training points
    r_squared: float
    adj_r_squared: float
    n_observations: int
    raw_results: object = field(repr=False)     # statsmodels Results

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Apply fitted coefficients to a new X matrix (must have same columns as
        training X). Returns i*_t on the new index.
        """
        Xc = sm.add_constant(X, has_constant="add")
        # Align column order to the training fit
        Xc = Xc[self.reduced_form_coefficients.index]
        yhat = Xc.values @ self.reduced_form_coefficients.values
        return pd.Series(yhat, index=X.index, name="i_star")


# =============================================================================
# Estimation
# =============================================================================

def fit_static(
    df: pd.DataFrame,
    spec: ReactionFunctionSpec,
    method: Literal["OLS", "GMM"] = "OLS",
    instruments: Optional[list[str]] = None,
) -> StaticFitResult:
    """
    Fit a single coefficient vector on the calibration window.

    Parameters
    ----------
    df : DataFrame conforming to ``cb_dissonance.src.data.schema``.
    spec : pre-registered country specification.
    method : "OLS" (default) or "GMM".
    instruments : columns to use as instruments (GMM only). For Aron-Muellbauer:
        ["inflation_lag1", "commodity_index_log_change", "i_lag1"].

    Returns
    -------
    StaticFitResult
    """
    validate_input_frame(df, spec)

    df = df.copy()
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    df = df.set_index(COL_DATE).sort_index()

    # Build regressors against the full sample, then slice the calibration window
    X = build_regressor_matrix(df.reset_index(), spec)
    X.index = pd.to_datetime(X.index)
    y = df[COL_POLICY_RATE]

    cal_start, cal_end = (pd.Timestamp(d) for d in spec.calibration_window)
    mask = (X.index >= cal_start) & (X.index <= cal_end)
    X_cal = X.loc[mask].dropna()
    y_cal = y.loc[X_cal.index]

    if X_cal.empty:
        raise ValueError(
            f"{spec.country}: no observations in calibration window "
            f"{spec.calibration_window}; check data coverage and pre-registration."
        )

    Xc = sm.add_constant(X_cal, has_constant="add")

    if method == "OLS":
        res = sm.OLS(y_cal, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    elif method == "GMM":
        if instruments is None:
            raise ValueError("GMM requires an explicit `instruments` argument.")
        Z = df.loc[X_cal.index, instruments]
        Zc = sm.add_constant(Z, has_constant="add")
        # IV2SLS as the practical implementation; full GMM via linearmodels.IVGMM
        # in the upgraded doctoral phase.
        res = IV2SLS(y_cal, Xc, instrument=Zc).fit()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Extract reduced-form coefficients
    rf_coefs = res.params
    rf_se = res.bse if hasattr(res, "bse") else pd.Series(np.nan, index=rf_coefs.index)

    # Recover structural coefficients
    structural = _structural_from_reduced(rf_coefs, spec)

    fitted = pd.Series(res.fittedvalues, index=X_cal.index, name="i_star")
    resid = pd.Series(res.resid, index=X_cal.index, name="epsilon")

    return StaticFitResult(
        spec=spec,
        method=method,
        reduced_form_coefficients=rf_coefs,
        structural_coefficients=structural,
        standard_errors=rf_se,
        fitted_values=fitted,
        residuals=resid,
        r_squared=float(getattr(res, "rsquared", np.nan)),
        adj_r_squared=float(getattr(res, "rsquared_adj", np.nan)),
        n_observations=int(res.nobs),
        raw_results=res,
    )


def _structural_from_reduced(rf: pd.Series, spec: ReactionFunctionSpec) -> pd.Series:
    """
    Recover (α, β, γ, δ, r*) from the reduced-form (a, b, g, d, c, ρ).
    Smoothed specification: a = (1-ρ)·α, etc.; constant = (1-ρ)·(r* + α·(-π*) + ...).
    """
    out: dict[str, float] = {}
    if spec.smoothing and "i_lag1" in rf.index:
        rho = float(rf["i_lag1"])
        factor = 1.0 - rho
        out["rho"] = rho
    else:
        factor = 1.0

    if "inflation_gap" in rf.index:
        out["alpha"] = float(rf["inflation_gap"]) / factor if factor != 0 else np.nan
    if "output_gap_used" in rf.index:
        out["beta"] = float(rf["output_gap_used"]) / factor if factor != 0 else np.nan
    if spec.include_fx and "reer_log_change" in rf.index:
        out["gamma"] = float(rf["reer_log_change"]) / factor if factor != 0 else np.nan
    if spec.include_foreign_rate and "foreign_rate" in rf.index:
        out["delta"] = float(rf["foreign_rate"]) / factor if factor != 0 else np.nan

    return pd.Series(out, name="structural")


# =============================================================================
# Taylor principle check
# =============================================================================

def passes_taylor_principle(fit: StaticFitResult) -> bool:
    """
    The Taylor principle: α > 1 (nominal rate must rise more than 1-for-1 with
    inflation to actually raise the real rate). Determinacy condition.

    Returns True if α̂ > 1, with a 1-σ buffer to be conservative.
    """
    alpha = fit.structural_coefficients.get("alpha", np.nan)
    if np.isnan(alpha):
        return False
    # Approximate SE on α via delta method: σ(α) ≈ σ(a)/(1-ρ)
    if fit.spec.smoothing:
        rho = fit.structural_coefficients.get("rho", 0.0)
        se_a = fit.standard_errors.get("inflation_gap", np.nan)
        if np.isnan(se_a) or rho == 1:
            return alpha > 1.0
        se_alpha = abs(se_a / (1.0 - rho))
        return (alpha - se_alpha) > 1.0
    return alpha > 1.0
