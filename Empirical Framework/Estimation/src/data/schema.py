"""
Canonical input data schema for L1 reaction function estimation.

Estimation code is decoupled from data sourcing. Every loader (Bloomberg,
Refinitiv, IMF IFS, BIS, national central bank websites) must produce a
DataFrame conforming to this schema before estimation modules touch it.

This separation lets the data layer absorb messiness (revisions, frequency
mismatches, proxy decisions) without leaking into the econometric code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# =============================================================================
# Column names — canonical
# =============================================================================

# Required for all countries
COL_DATE = "date"                          # datetime, period-end
COL_COUNTRY = "country"                    # ISO-3 code
COL_POLICY_RATE = "policy_rate"            # i_t, % annualised
COL_INFLATION = "inflation"                # π_t, % annualised YoY
COL_INFLATION_TARGET = "pi_star"           # π*, % annualised (may vary over time)
COL_OUTPUT_GAP = "output_gap"              # ỹ_t, % of potential GDP
COL_INFLATION_EXP_H = "inflation_exp_h4"   # E_t π_{t+h}, h=4 quarters default
COL_OUTPUT_GAP_EXP = "output_gap_exp_k"    # E_t ỹ_{t+k}

# Required for EM (Aron-Muellbauer / Mohanty-Klau specification)
COL_REER_CHANGE = "reer_log_change"        # Δlog(REER), monthly
COL_FOREIGN_RATE = "foreign_rate"          # i^foreign_t, typically USFFR

# Intensity / instrumental variables
COL_COMMODITY_INDEX = "commodity_index_log_change"  # for GMM instrument set
COL_LAGGED_INFLATION = "inflation_lag1"             # for instrument set

# Bookkeeping
COL_REAL_TIME = "is_real_time"  # bool — True if real-time vintage, False if revised


# =============================================================================
# Spec / validation
# =============================================================================

@dataclass(frozen=True)
class ReactionFunctionSpec:
    """
    Single source of truth for a country's reaction function specification.
    Loaded from ``config/preregistration.yaml`` — do not construct ad-hoc.
    """
    country: str                           # ISO-3
    spec_type: str                         # "CGG_DM" | "AM_MK_EM" | "TAYLOR_BASIC"
    pi_star: float                         # if constant; if time-varying, use pi_star series in data
    r_star: float                          # neutral real rate prior
    inflation_horizon: int                 # h in E_t π_{t+h}; 0 → use realised
    output_gap_horizon: int                # k in E_t ỹ_{t+k}; 0 → use realised
    smoothing: bool                        # include ρ·i_{t-1}
    include_fx: bool
    include_foreign_rate: bool
    calibration_window: tuple[str, str]    # ("YYYY-MM-DD", "YYYY-MM-DD")
    test_window: tuple[str, str]
    frequency: str = "M"                   # "M" monthly, "Q" quarterly
    panel: str = "EM"                      # "DM" | "EM"
    notes: str = ""

    @property
    def regressors(self) -> list[str]:
        """Column names the rule uses as regressors (excluding the constant)."""
        cols = []
        if self.smoothing:
            cols.append("i_lag1")
        cols.append("inflation_gap")          # E_t π_{t+h} − π*
        cols.append("output_gap_used")        # E_t ỹ_{t+k} or ỹ_t
        if self.include_fx:
            cols.append(COL_REER_CHANGE)
        if self.include_foreign_rate:
            cols.append(COL_FOREIGN_RATE)
        return cols


# =============================================================================
# Validation
# =============================================================================

class SchemaError(ValueError):
    """Raised when input data fails schema validation."""


def validate_input_frame(df: pd.DataFrame, spec: ReactionFunctionSpec) -> None:
    """
    Validate that a DataFrame meets the requirements for the given spec.
    Fails loudly: any silent coercion masks data problems we want to surface.
    """
    required = [COL_DATE, COL_POLICY_RATE, COL_INFLATION, COL_OUTPUT_GAP]

    # Inflation target may be a constant (in spec) or a column (when time-varying)
    if COL_INFLATION_TARGET not in df.columns and spec.pi_star is None:
        raise SchemaError(
            f"{spec.country}: pi_star not in spec and {COL_INFLATION_TARGET!r} not in data"
        )

    # Forward-looking specs need an expectations series; otherwise we fall back to realised
    if spec.inflation_horizon > 0 and COL_INFLATION_EXP_H not in df.columns:
        raise SchemaError(
            f"{spec.country}: forward-looking spec (h={spec.inflation_horizon}) "
            f"but {COL_INFLATION_EXP_H!r} not in data. "
            f"Set inflation_horizon=0 to use realised, or provide expectations series."
        )

    if spec.include_fx and COL_REER_CHANGE not in df.columns:
        raise SchemaError(f"{spec.country}: include_fx=True but {COL_REER_CHANGE!r} not in data")
    if spec.include_foreign_rate and COL_FOREIGN_RATE not in df.columns:
        raise SchemaError(f"{spec.country}: include_foreign_rate=True but {COL_FOREIGN_RATE!r} not in data")

    for col in required:
        if col not in df.columns:
            raise SchemaError(f"{spec.country}: required column {col!r} missing")

    # Date column must be sorted and unique
    if not df[COL_DATE].is_monotonic_increasing:
        raise SchemaError(f"{spec.country}: {COL_DATE} not monotonic increasing")
    if df[COL_DATE].duplicated().any():
        raise SchemaError(f"{spec.country}: duplicate dates in input")


def build_regressor_matrix(df: pd.DataFrame, spec: ReactionFunctionSpec) -> pd.DataFrame:
    """
    Construct the regressor matrix X used by every estimator. Derived columns:
      - i_lag1:          policy rate lagged once
      - inflation_gap:   E_t π_{t+h} − π*   (or π_t − π* if h=0)
      - output_gap_used: E_t ỹ_{t+k}        (or ỹ_t if k=0)

    The constant term is added by the estimator (statsmodels.add_constant), not here.
    """
    out = pd.DataFrame(index=df[COL_DATE].values)

    if spec.smoothing:
        out["i_lag1"] = df[COL_POLICY_RATE].shift(1).values

    # Inflation gap
    pi_star = (
        df[COL_INFLATION_TARGET].values
        if COL_INFLATION_TARGET in df.columns
        else spec.pi_star
    )
    if spec.inflation_horizon > 0:
        pi_used = df[COL_INFLATION_EXP_H].values
    else:
        pi_used = df[COL_INFLATION].values
    out["inflation_gap"] = pi_used - pi_star

    # Output gap (expected if specified, else realised)
    if spec.output_gap_horizon > 0 and COL_OUTPUT_GAP_EXP in df.columns:
        out["output_gap_used"] = df[COL_OUTPUT_GAP_EXP].values
    else:
        out["output_gap_used"] = df[COL_OUTPUT_GAP].values

    if spec.include_fx:
        out[COL_REER_CHANGE] = df[COL_REER_CHANGE].values
    if spec.include_foreign_rate:
        out[COL_FOREIGN_RATE] = df[COL_FOREIGN_RATE].values

    out.index.name = COL_DATE
    return out
