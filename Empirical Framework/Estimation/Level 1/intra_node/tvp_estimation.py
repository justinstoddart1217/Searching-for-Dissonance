"""
Time-Varying Parameter (TVP) estimation of the reaction function.

State equation:    θ_t = θ_{t−1} + η_t,    η_t ~ N(0, Q)
Observation eq.:   i_t = X_t' θ_t + ε_t,   ε_t ~ N(0, σ²)

Estimated via Kalman filter + RTS smoother (statsmodels MLEModel). The headline
intra-L1 dissonance is built on the filtered (real-time) coefficient path so
that the metric reflects what an observer could have inferred at time t. The
smoothed path is also retained, for descriptive plotting and for the data
appendix.

Design choices:
  - Q is parameterised as a diagonal matrix `q_i = exp(2·log_q_i)`. Each
    coefficient gets its own innovation variance. ML estimates these jointly
    with σ²_ε.
  - The state vector includes the constant ("const") as the first element;
    intercept drift is allowed because r* itself is not stable.
  - Initialisation: diffuse prior on the state, except where the static fit
    gives a strong informative starting value — then we initialise the filter
    at the static θ̂ to anchor the early sample.

The two consumers of this fit:
  - intra-L1 dissonance: ||η_t|| or per-coefficient |η_t,j|
  - L1↔L3a vertical dissonance: |i_t − X_t' θ_t| evaluated using the TVP path

These share the same underlying ``TVPFitResult`` object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.mlemodel import MLEModel

from cb_dissonance.src.data.schema import (
    ReactionFunctionSpec,
    COL_DATE,
    COL_POLICY_RATE,
    build_regressor_matrix,
    validate_input_frame,
)


# =============================================================================
# State-space model
# =============================================================================

class _TVPReactionFunctionModel(MLEModel):
    """
    State-space form:
      y_t   = Z_t · α_t + ε_t,   ε_t ~ N(0, h)
      α_t   = α_{t-1} + η_t,     η_t ~ N(0, Q)
    where Z_t is the row of regressors at t (1 x p), α_t is the coefficient
    vector (p x 1), Q is diagonal p x p, h scalar observation variance.
    """

    def __init__(self, endog: np.ndarray, exog: np.ndarray, initial_state: np.ndarray):
        # Number of states = number of regressors (including constant)
        self.k_states_internal = exog.shape[1]
        super().__init__(
            endog=endog,
            k_states=self.k_states_internal,
            k_posdef=self.k_states_internal,
            initialization="known",
            initial_state=initial_state,
            initial_state_cov=np.eye(self.k_states_internal) * 1e2,
        )
        self._exog = exog
        self._param_names = (
            [f"log_q_{j}" for j in range(self.k_states_internal)] + ["log_h"]
        )

        # Time-varying Z (design matrix) — shape (k_endog, k_states, nobs)
        self["design"] = np.zeros((1, self.k_states_internal, self.nobs))
        for t in range(self.nobs):
            self["design", 0, :, t] = exog[t, :]

        self["transition"] = np.eye(self.k_states_internal)
        self["selection"] = np.eye(self.k_states_internal)

    @property
    def param_names(self) -> list[str]:
        return self._param_names

    @property
    def start_params(self) -> np.ndarray:
        # Conservative starting values: small state innovations, observation
        # variance from OLS residuals
        return np.array([np.log(1e-4)] * self.k_states_internal + [np.log(0.1)])

    def update(self, params: np.ndarray, **kwargs) -> None:
        params = super().update(params, **kwargs)
        q_diag = np.exp(2 * params[: self.k_states_internal])
        h = np.exp(2 * params[self.k_states_internal])
        self["state_cov"] = np.diag(q_diag)
        self["obs_cov"] = np.array([[h]])

    def transform_params(self, unconstrained):
        return unconstrained

    def untransform_params(self, constrained):
        return constrained


# =============================================================================
# Public API
# =============================================================================

@dataclass
class TVPFitResult:
    """Holds the full Kalman / smoother output and exposes downstream metrics."""

    spec: ReactionFunctionSpec
    filtered_reduced_form: pd.DataFrame      # θ_t|t  (reduced-form coefficients)
    smoothed_reduced_form: pd.DataFrame      # θ_t|T  (smoothed)
    filtered_structural: pd.DataFrame        # α_t, β_t, ρ_t, γ_t, δ_t  (filtered)
    smoothed_structural: pd.DataFrame        # … (smoothed)
    innovations_eta: pd.DataFrame            # η_t = θ_t|t − θ_{t-1}|t-1 in state space
    one_step_residuals: pd.Series            # ε_t  (forecast errors)
    fitted_values: pd.Series                 # i*_t = X_t' θ_t|t   (filtered)
    state_cov_diag: pd.DataFrame             # diag of cov(θ_t|t) — for confidence bands
    estimated_q_diag: pd.Series              # ML-estimated innovation variances
    estimated_h: float                       # ML-estimated obs variance
    loglik: float
    raw_results: object = field(repr=False)


def fit_tvp(
    df: pd.DataFrame,
    spec: ReactionFunctionSpec,
    initial_state: Optional[np.ndarray] = None,
    fixed_q_diag: Optional[np.ndarray] = None,
    fixed_h: Optional[float] = None,
    method: str = "lbfgs",
    maxiter: int = 200,
) -> TVPFitResult:
    """
    Fit the TVP reaction function over the full available sample.

    Parameters
    ----------
    df : input data.
    spec : pre-registered country spec.
    initial_state : optional initialisation for θ_0. Recommended: pass the
        StaticFitResult.reduced_form_coefficients values aligned to the regressor
        order. If None, defaults to zeros (diffuse prior).
    fixed_q_diag : optional diagonal of the state innovation covariance Q.
        If provided, Q is *not* ML-estimated; the filter is run with this Q held
        fixed. This is the practical fix for the "pile-up at zero" problem in
        state-space ML estimation, where the likelihood prefers Q ≈ 0 (treating
        coefficients as constant) and the filter fails to track real drift.
        Reasonable starting point: ``q_multiplier * np.var(static_residuals) *
        np.ones(k_states)`` with q_multiplier ~ 1e-3. Tune to match the empirical
        smoothness you observe in the rolling-window comparator.
    fixed_h : optional observation variance. If None, ML-estimated even when
        fixed_q_diag is provided.
    method : optimiser for ML (used only when fixed_q_diag is None).
    maxiter : optimiser iteration cap.

    Returns
    -------
    TVPFitResult
    """
    validate_input_frame(df, spec)

    df = df.copy()
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    df = df.set_index(COL_DATE).sort_index()

    X = build_regressor_matrix(df.reset_index(), spec)
    X.index = pd.to_datetime(X.index)
    y = df[COL_POLICY_RATE]

    full = pd.concat([y.rename("y"), X], axis=1).dropna()
    y_vec = full["y"].values
    X_mat = sm.add_constant(full.drop(columns=["y"]), has_constant="add").values
    col_order = ["const"] + list(full.drop(columns=["y"]).columns)
    dates = full.index

    if initial_state is None:
        initial_state = np.zeros(X_mat.shape[1])
    else:
        if len(initial_state) != X_mat.shape[1]:
            raise ValueError(
                f"initial_state length {len(initial_state)} != n_regressors {X_mat.shape[1]}"
            )

    if fixed_q_diag is not None and len(fixed_q_diag) != X_mat.shape[1]:
        raise ValueError(
            f"fixed_q_diag length {len(fixed_q_diag)} != n_regressors {X_mat.shape[1]}"
        )

    model = _TVPReactionFunctionModel(endog=y_vec, exog=X_mat, initial_state=initial_state)

    if fixed_q_diag is not None:
        # Skip ML; install Q (and optionally h) directly and run the filter once
        h_val = fixed_h if fixed_h is not None else float(np.var(y_vec) * 0.05)
        params = np.concatenate([
            0.5 * np.log(fixed_q_diag),  # log of sqrt(Q) since model uses exp(2·log)
            [0.5 * np.log(h_val)],
        ])
        res = model.smooth(params)
        llf = float(res.llf)
    else:
        res = model.fit(method=method, maxiter=maxiter, disp=False)
        llf = float(res.llf)

    filt = pd.DataFrame(res.filtered_state.T, index=dates, columns=col_order)
    smoo = pd.DataFrame(res.smoothed_state.T, index=dates, columns=col_order)

    fsc = res.filtered_state_cov
    cov_diag = np.array([np.diag(fsc[:, :, t]) for t in range(fsc.shape[2])])
    cov_df = pd.DataFrame(cov_diag, index=dates, columns=col_order)

    eta = filt.diff().rename(columns=lambda c: f"eta_{c}")

    filt_struct = _structural_from_tvp(filt, spec)
    smoo_struct = _structural_from_tvp(smoo, spec)

    fitted = (X_mat * filt.values).sum(axis=1)
    fitted_s = pd.Series(fitted, index=dates, name="i_star_tvp")
    resid_s = pd.Series(y_vec - fitted, index=dates, name="epsilon_tvp")

    if fixed_q_diag is not None:
        q_hat = np.asarray(fixed_q_diag, dtype=float)
        h_hat = float(fixed_h) if fixed_h is not None else float(np.var(y_vec) * 0.05)
    else:
        q_hat = np.exp(2 * res.params[: model.k_states_internal])
        h_hat = float(np.exp(2 * res.params[model.k_states_internal]))

    return TVPFitResult(
        spec=spec,
        filtered_reduced_form=filt,
        smoothed_reduced_form=smoo,
        filtered_structural=filt_struct,
        smoothed_structural=smoo_struct,
        innovations_eta=eta,
        one_step_residuals=resid_s,
        fitted_values=fitted_s,
        state_cov_diag=cov_df,
        estimated_q_diag=pd.Series(q_hat, index=col_order, name="Q_diag"),
        estimated_h=h_hat,
        loglik=llf,
        raw_results=res,
    )


def _structural_from_tvp(rf_df: pd.DataFrame, spec: ReactionFunctionSpec) -> pd.DataFrame:
    """
    Vectorised recovery of structural (α, β, ρ, γ, δ) from the time-varying
    reduced form. Element-wise division by (1−ρ_t) when smoothing is on.
    """
    out = pd.DataFrame(index=rf_df.index)
    if spec.smoothing and "i_lag1" in rf_df.columns:
        rho = rf_df["i_lag1"]
        factor = 1.0 - rho
        out["rho"] = rho
    else:
        factor = pd.Series(1.0, index=rf_df.index)

    # Guard against ρ ≈ 1 (factor → 0)
    factor = factor.where(factor.abs() > 1e-4)

    if "inflation_gap" in rf_df.columns:
        out["alpha"] = rf_df["inflation_gap"] / factor
    if "output_gap_used" in rf_df.columns:
        out["beta"] = rf_df["output_gap_used"] / factor
    if spec.include_fx and "reer_log_change" in rf_df.columns:
        out["gamma"] = rf_df["reer_log_change"] / factor
    if spec.include_foreign_rate and "foreign_rate" in rf_df.columns:
        out["delta"] = rf_df["foreign_rate"] / factor
    return out
