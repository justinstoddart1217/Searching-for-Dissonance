"""
End-to-end pilot for L1 intra-node dissonance.

Generates a synthetic 2005-2024 monthly series with a deliberately injected
coefficient regime shift around 2018 — α drifts from 1.5 (Taylor-principle
hawk) to 0.9 (passive). The pipeline should detect this drift via the TVP fit
and fire dissonance after the shift, while remaining quiet over the
calibration window 2005-2014.

Run as: python -m cb_dissonance.notebooks.01_l1_intra_pilot

This is the validation harness. Once it passes, swap the synthetic loader for
the real Bloomberg/Refinitiv loader and re-run on SA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
from cb_dissonance.src.level_1.intra_node.diagnostics import run_l1_intra_pipeline
from cb_dissonance.src.level_1.intra_node.rolling_estimation import fit_rolling


# =============================================================================
# Synthetic data generator
# =============================================================================

def generate_synthetic_sa(seed: int = 42) -> pd.DataFrame:
    """
    Synthetic South Africa-like monthly series, 2005-01 to 2024-12.

    Built so that:
      - α (inflation coefficient) drifts from 1.5 → 0.9 starting 2018-01.
        This is the canonical "rule degrading" case — the bank is no longer
        responding strongly enough to inflation to satisfy the Taylor principle.
      - β (output gap) ~ 0.5 throughout.
      - ρ (smoothing) ~ 0.8 throughout.
      - γ (REER) ~ -0.05 throughout.
      - δ (foreign rate) ~ 0.3 throughout.

    The intra-L1 dissonance metric should:
      - Sit near zero pre-2018.
      - Rise after 2018 as α_t pulls away from the calibration baseline.
      - Cross τ somewhere in 2019-2020 depending on TVP smoothness.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-31", "2024-12-31", freq="ME")
    T = len(dates)

    # Exogenous state
    pi_star = 4.5
    inflation = 4.5 + np.cumsum(rng.normal(0, 0.15, T))
    inflation = np.clip(inflation, 1.0, 12.0)
    output_gap = np.cumsum(rng.normal(0, 0.10, T))
    output_gap = np.clip(output_gap, -4.0, 4.0)
    reer_change = rng.normal(0, 0.02, T)
    us_ffr = np.clip(2.0 + np.cumsum(rng.normal(0, 0.10, T)), 0.0, 6.0)

    # Inflation expectations: rational with small noise around 4Q-ahead realisation
    infl_exp = pd.Series(inflation).shift(-4).ffill().values + rng.normal(0, 0.2, T)

    # Time-varying coefficients (the regime shift sits in α)
    alpha = np.where(np.arange(T) < 156, 1.5, 0.9)  # 2005-01 to 2017-12 then drop
    # Smooth the transition over 12 months so it's not a knife-edge
    transition_start = 156
    for k in range(12):
        if transition_start + k < T:
            alpha[transition_start + k] = 1.5 - (1.5 - 0.9) * (k / 12)

    beta = np.full(T, 0.5)
    rho = np.full(T, 0.8)
    gamma = np.full(T, -0.05)
    delta = np.full(T, 0.3)

    # Simulate the policy rate forward
    i = np.zeros(T)
    i[0] = 7.0  # SARB c.2005
    sigma_eps = 0.15
    for t in range(1, T):
        target = (
            2.0 + pi_star
            + alpha[t] * (infl_exp[t] - pi_star)
            + beta[t] * output_gap[t]
            + gamma[t] * reer_change[t] * 100  # scale for visibility
            + delta[t] * us_ffr[t]
        )
        i[t] = rho[t] * i[t - 1] + (1 - rho[t]) * target + rng.normal(0, sigma_eps)
    i = np.clip(i, 0.5, 18.0)

    return pd.DataFrame({
        COL_DATE: dates,
        "country": "ZAF",
        COL_POLICY_RATE: i,
        COL_INFLATION: inflation,
        COL_INFLATION_TARGET: pi_star,
        COL_OUTPUT_GAP: output_gap,
        COL_INFLATION_EXP_H: infl_exp,
        COL_REER_CHANGE: reer_change,
        COL_FOREIGN_RATE: us_ffr,
    })


# =============================================================================
# Spec — would normally come from preregistration.yaml
# =============================================================================

SA_SPEC = ReactionFunctionSpec(
    country="ZAF",
    spec_type="AM_MK_EM",
    pi_star=4.5,
    r_star=2.0,
    inflation_horizon=4,
    output_gap_horizon=0,
    smoothing=True,
    include_fx=True,
    include_foreign_rate=True,
    calibration_window=("2005-01-01", "2014-12-31"),
    test_window=("2015-01-01", "2024-12-31"),
    frequency="M",
    panel="EM",
)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("L1 INTRA-NODE PILOT — synthetic SA, α regime shift 2018")
    print("=" * 70)

    df = generate_synthetic_sa()
    test_window_data = df[
        (df[COL_DATE] >= SA_SPEC.test_window[0])
        & (df[COL_DATE] <= SA_SPEC.test_window[1])
    ].copy()

    results = run_l1_intra_pipeline(df, SA_SPEC, test_window_data=test_window_data)

    static = results["static_fit"]
    intensity = results["intensity"]
    tvp = results["tvp_fit"]
    diss = results["dissonance"]

    # --- Static fit summary ----------------------------------------------------
    print("\n[1] STATIC REACTION FUNCTION  (calibration window 2005-2014)")
    print("-" * 70)
    print(f"  Method:          {static.method}")
    print(f"  R²:              {static.r_squared:.4f}")
    print(f"  Adj R²:          {static.adj_r_squared:.4f}")
    print(f"  N observations:  {static.n_observations}")
    print(f"  Structural coefficients:")
    for k, v in static.structural_coefficients.items():
        print(f"    {k:12s}  =  {v:8.4f}")

    # --- Intensity ------------------------------------------------------------
    print("\n[2] INTENSITY TEST")
    print("-" * 70)
    print(f"  R²:                  {intensity.r_squared:.4f}  "
          f"(threshold 0.70, {'PASS' if intensity.passes_r_squared else 'FAIL'})")
    print(f"  σ(ε)/σ(i):           {intensity.residual_std_ratio:.4f}  "
          f"(threshold 0.40, {'PASS' if intensity.passes_residual_ratio else 'FAIL'})")
    if intensity.out_of_sample_rmse is not None:
        print(f"  OOS RMSE (test):     {intensity.out_of_sample_rmse:.4f}")
    print(f"  Granger p-values:")
    for k, v in intensity.granger_p_values.items():
        flag = "PASS" if (not np.isnan(v) and v < 0.05) else "—"
        print(f"    {k:25s}  p = {v:.4f}   {flag}")
    print(f"  SUFFICIENCY:   {'PASS' if intensity.passes_sufficiency else 'FAIL'}")

    # --- TVP fit summary ------------------------------------------------------
    print("\n[3] TVP REACTION FUNCTION")
    print("-" * 70)
    print(f"  Log-likelihood:   {tvp.loglik:.2f}")
    print(f"  Estimated Q (innovation variances):")
    for k, v in tvp.estimated_q_diag.items():
        print(f"    Q[{k:15s}]  =  {v:.6e}")
    print(f"  Estimated h (obs variance):  {tvp.estimated_h:.6f}")
    print(f"\n  Filtered α (inflation coef.) at key dates:")
    alpha_series = tvp.filtered_structural["alpha"]
    for date_str in ["2008-12-31", "2014-12-31", "2018-06-30", "2019-12-31",
                     "2022-12-31", "2024-12-31"]:
        ts = pd.Timestamp(date_str)
        try:
            # Find nearest available date
            nearest = alpha_series.index[alpha_series.index.get_indexer([ts], method="nearest")[0]]
            print(f"    {nearest.date()}   α  =  {alpha_series.loc[nearest]:.4f}")
        except (KeyError, IndexError):
            pass

    # --- Dissonance -----------------------------------------------------------
    print("\n[4] INTRA-L1 DISSONANCE")
    print("-" * 70)
    print(f"  Headline threshold τ (2σ):  {diss.threshold:.4f}")
    print(f"  Threshold robustness:")
    for k, v in diss.threshold_robustness.items():
        print(f"    {k:10s}  τ = {v:.4f}")
    n_fires = int(diss.fires.sum())
    print(f"  N periods firing:  {n_fires} / {len(diss.fires)}  "
          f"({100*n_fires/len(diss.fires):.1f}%)")

    # Where do fires concentrate?
    print(f"\n  Periods with firing (showing first 10):")
    firing = diss.fires[diss.fires]
    for d in firing.index[:10]:
        d_val = diss.d_mahalanobis.loc[d]
        print(f"    {d.date()}   D = {d_val:.4f}")
    if len(firing) > 10:
        print(f"    … plus {len(firing) - 10} more")

    print(f"\n  Per-coefficient persistence (AR(1) of |drift|):")
    for k, v in diss.persistence_per_coef.items():
        print(f"    {k:12s}  ρ_AR1 = {v:.4f}")

    # --- Rolling-window comparator --------------------------------------------
    print("\n[5] ROLLING-WINDOW COMPARATOR  (60m window)")
    print("-" * 70)
    rolling = fit_rolling(df, SA_SPEC, window=60)
    print(f"  N windows fit:  {len(rolling.structural_coefficients)}")
    print(f"  R² range:       [{rolling.r_squared.min():.3f}, {rolling.r_squared.max():.3f}]")
    print(f"  α range across windows: "
          f"[{rolling.structural_coefficients['alpha'].min():.3f}, "
          f"{rolling.structural_coefficients['alpha'].max():.3f}]")

    print("\n" + "=" * 70)
    print("PILOT COMPLETE.  Next step: replace generate_synthetic_sa with the")
    print("real SA loader and re-run.")
    print("=" * 70)


if __name__ == "__main__":
    main()
