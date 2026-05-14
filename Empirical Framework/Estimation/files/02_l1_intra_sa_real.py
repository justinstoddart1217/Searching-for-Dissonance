"""
End-to-end L1 intra-node pipeline on real South Africa data.

This is notebook 01's structure but with the synthetic data generator
replaced by ``load_from_csv``. Once you populate the CSV (or assemble
from a series dict), the rest of the pipeline runs unchanged.

Two workflows shown:
  - Workflow A: pre-aligned monthly CSV, one column per canonical series
  - Workflow B: each variable from a separate pd.Series, with frequency
    alignment handled inside the loader

Use whichever fits your data sourcing. The estimation code does not care.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cb_dissonance.src.data.loaders import (
    load_from_csv,
    load_from_series_dict,
    write_template_csv,
)
from cb_dissonance.src.data.transforms import (
    hp_filter_output_gap,
    reer_log_change,
    align_to_monthly,
)
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
from cb_dissonance.src.level_1.intra_node.reaction_function import fit_static
from cb_dissonance.src.level_1.intra_node.diagnostics import run_l1_intra_pipeline


# =============================================================================
# Spec — loaded from preregistration.yaml in production. Inline here for clarity.
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
# Workflow A: pre-aligned monthly CSV
# =============================================================================

def workflow_a_csv() -> pd.DataFrame:
    """
    Load from a single wide CSV where each column is one canonical series.

    Typical: export from Bloomberg via BDH / BQL into Excel, save as CSV,
    drop in data/raw/sa/sa_panel.csv. Columns can have raw ticker names
    (use ``column_map``) or the canonical names already.
    """
    csv_path = Path("cb_dissonance/data/raw/sa/sa_panel.csv")

    if not csv_path.exists():
        # First run: write an empty template
        write_template_csv(csv_path, start="2005-01-31", end="2024-12-31", panel="EM")
        print()
        print("Template written. Populate it from your data sources, then re-run.")
        print("Required columns:", [
            COL_POLICY_RATE, COL_INFLATION, COL_INFLATION_TARGET,
            COL_OUTPUT_GAP, COL_INFLATION_EXP_H, COL_REER_CHANGE, COL_FOREIGN_RATE,
        ])
        return None

    df = load_from_csv(
        csv_path,
        country="ZAF",
        # If your CSV has Bloomberg tickers as column names, map them here:
        column_map={
            # "SARRRD Index": COL_POLICY_RATE,
            # "SACPYOY Index": COL_INFLATION,
            # "BER_1Y_INFL_EXP": COL_INFLATION_EXP_H,
            # "SBZAREER Index": COL_REER_CHANGE,
            # "FEDL01 Index": COL_FOREIGN_RATE,
        },
    )
    return df


# =============================================================================
# Workflow B: series-by-series, mixed frequencies
# =============================================================================

def workflow_b_series_dict(
    repo_rate_daily: pd.Series,
    cpi_monthly: pd.Series,
    real_gdp_quarterly: pd.Series,
    ber_exp_quarterly: pd.Series,
    reer_monthly: pd.Series,
    fed_funds_daily: pd.Series,
) -> pd.DataFrame:
    """
    Assemble the panel when each variable comes from a different source at
    a different frequency. Each input is a pd.Series with a DatetimeIndex.

    Derivations applied:
      - policy rate:        align daily → monthly (last)
      - inflation:          identity (already monthly YoY)
      - output gap:         HP filter on log GDP, quarterly → monthly (ffill)
      - inflation exp:      quarterly survey → monthly (ffill)
      - REER change:        Δlog from REER level
      - foreign rate:       align daily → monthly (mean)
    """
    log_gdp = np.log(real_gdp_quarterly)
    output_gap = hp_filter_output_gap(log_gdp, lamb=1600)

    series = {
        COL_POLICY_RATE: align_to_monthly(repo_rate_daily, method="last"),
        COL_INFLATION: cpi_monthly,
        COL_OUTPUT_GAP: output_gap,
        COL_INFLATION_EXP_H: ber_exp_quarterly,
        COL_REER_CHANGE: reer_log_change(reer_monthly),
        COL_FOREIGN_RATE: align_to_monthly(fed_funds_daily, method="mean"),
    }
    df = load_from_series_dict(series, country="ZAF", target_frequency="M")
    df[COL_INFLATION_TARGET] = 4.5  # SARB point target
    return df


# =============================================================================
# Pipeline runner
# =============================================================================

def run(df: pd.DataFrame) -> None:
    """Run intensity → static → rolling → TVP → dissonance, same as notebook 01."""
    static = fit_static(df, SA_SPEC, method="OLS")
    print("\n[Static fit] structural coefficients:")
    print(static.structural_coefficients.round(3))
    print(f"R² = {static.r_squared:.3f}")

    result = run_l1_intra_pipeline(
        df=df,
        spec=SA_SPEC,
        # For real data: pre-register fixed_q_diag rather than relying on ML
        # to avoid the Q pile-up problem documented in tvp_estimation.py
        tvp_fixed_q_diag=np.array([1e-3] * len(static.reduced_form_coefficients)),
        tvp_fixed_h=None,  # let ML estimate observation variance
    )

    print("\n[Intensity report]")
    print(f"  R² = {result.intensity.r_squared:.3f} (passes: {result.intensity.passes_r_squared})")
    print(f"  σ ratio = {result.intensity.residual_std_ratio:.3f} (passes: {result.intensity.passes_residual_ratio})")
    print(f"  Granger min p = {min(result.intensity.granger_p_values.values()):.4f} (passes: {result.intensity.passes_granger})")
    print(f"  Sufficiency: {result.intensity.passes_sufficiency}")

    print("\n[Dissonance trace]")
    D = result.dissonance.d_mahalanobis.dropna()
    fires = result.dissonance.fires.dropna()
    print(f"  threshold τ = {result.dissonance.threshold:.3f}")
    print(f"  firing rate (full sample) = {fires.mean():.1%}")
    test_fires = fires.loc[SA_SPEC.test_window[0]:]
    print(f"  firing rate (test window) = {test_fires.mean():.1%}")


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    df = workflow_a_csv()
    if df is None:
        # Template was just written — exit and wait for population
        raise SystemExit
    run(df)
