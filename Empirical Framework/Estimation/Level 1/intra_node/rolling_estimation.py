"""
Rolling-window reaction function estimation.

Re-fits the static reduced form on a sliding window of width ``window`` periods
and returns θ̂_t as a DataFrame indexed by date. This is the *transparent*
comparator to the TVP fit:

  - Rolling: no state-space assumptions, no smoothing across windows, no priors.
    Easier to defend in a viva because the only choice is the window width.
  - TVP: smoother, more efficient, but conditional on the Kalman filter's
    assumption that θ_t follows a random walk with covariance Q.

If the two estimators disagree on the *direction* of drift, the dissonance
signal is fragile. If they agree on direction and disagree only on magnitude,
TVP carries the headline and rolling is reported as robustness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cb_dissonance.src.data.schema import (
    ReactionFunctionSpec,
    COL_DATE,
    COL_POLICY_RATE,
    build_regressor_matrix,
    validate_input_frame,
)


@dataclass
class RollingFitResult:
    spec: ReactionFunctionSpec
    window: int
    min_periods: int
    reduced_form_coefficients: pd.DataFrame   # rows = window-end date, cols = coefficient
    structural_coefficients: pd.DataFrame     # α, β, ρ, γ, δ over time
    standard_errors: pd.DataFrame             # HAC SEs on reduced form
    fitted_values: pd.Series                  # i*_t evaluated with the θ̂ of the window ending at t
    residuals: pd.Series                      # i_t − i*_t (in-sample for the trailing window)
    r_squared: pd.Series                      # R² per window


def fit_rolling(
    df: pd.DataFrame,
    spec: ReactionFunctionSpec,
    window: int = 60,
    min_periods: Optional[int] = None,
) -> RollingFitResult:
    """
    Fit the reaction function on rolling windows.

    Parameters
    ----------
    df : input data conforming to schema.
    spec : pre-registered country specification.
    window : window width in periods (monthly: 60 = 5 years).
    min_periods : minimum periods to fit; defaults to window. Setting this lower
        produces an expanding-then-rolling pattern at the start of the sample.
    """
    if min_periods is None:
        min_periods = window
    validate_input_frame(df, spec)

    df = df.copy()
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    df = df.set_index(COL_DATE).sort_index()

    X = build_regressor_matrix(df.reset_index(), spec)
    X.index = pd.to_datetime(X.index)
    y = df[COL_POLICY_RATE]

    full = pd.concat([y.rename("y"), X], axis=1).dropna()

    rf_rows: list[pd.Series] = []
    se_rows: list[pd.Series] = []
    struct_rows: list[pd.Series] = []
    r2 = []
    fitted_at_end: list[float] = []
    resid_at_end: list[float] = []
    end_dates: list[pd.Timestamp] = []

    n = len(full)
    for end in range(min_periods, n + 1):
        start = max(0, end - window)
        sub = full.iloc[start:end]
        y_sub = sub["y"]
        X_sub = sub.drop(columns=["y"])
        Xc = sm.add_constant(X_sub, has_constant="add")
        try:
            res = sm.OLS(y_sub, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
        except Exception:
            continue
        rf = res.params.rename(sub.index[-1])
        se = res.bse.rename(sub.index[-1])
        rf_rows.append(rf)
        se_rows.append(se)
        struct_rows.append(_structural_from_reduced(rf, spec).rename(sub.index[-1]))
        r2.append((sub.index[-1], float(res.rsquared)))
        fitted_at_end.append(float(res.fittedvalues.iloc[-1]))
        resid_at_end.append(float(res.resid.iloc[-1]))
        end_dates.append(sub.index[-1])

    rf_df = pd.DataFrame(rf_rows)
    se_df = pd.DataFrame(se_rows)
    struct_df = pd.DataFrame(struct_rows)
    r2_s = pd.Series(dict(r2)).rename("r_squared")
    fitted_s = pd.Series(fitted_at_end, index=pd.Index(end_dates, name=COL_DATE), name="i_star")
    resid_s = pd.Series(resid_at_end, index=pd.Index(end_dates, name=COL_DATE), name="epsilon")

    return RollingFitResult(
        spec=spec,
        window=window,
        min_periods=min_periods,
        reduced_form_coefficients=rf_df,
        structural_coefficients=struct_df,
        standard_errors=se_df,
        fitted_values=fitted_s,
        residuals=resid_s,
        r_squared=r2_s,
    )


def _structural_from_reduced(rf: pd.Series, spec: ReactionFunctionSpec) -> pd.Series:
    """Same recovery logic as in reaction_function._structural_from_reduced, vectorisable."""
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
