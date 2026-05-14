"""
Common transformations from raw source data → canonical schema columns.

The output gap is rarely available directly; it must be derived from real
GDP. REER changes are derived from REER levels. Inflation expectations
come at lower frequency than the policy rate and need alignment. These
helpers do the standard transformations; the choice of which to apply per
country is documented in ``config/data_manifest_<country>.yaml``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


# =============================================================================
# HP filter for output gap
# =============================================================================

def hp_filter_output_gap(
    log_gdp: pd.Series,
    lamb: float = 1600.0,
    one_sided: bool = False,
) -> pd.Series:
    """
    Derive the output gap from log real GDP via the Hodrick-Prescott filter.

    Parameters
    ----------
    log_gdp : pd.Series of log(real GDP), DatetimeIndex (typically quarterly).
    lamb : smoothing parameter. 1600 is standard for quarterly data; 14400
        for monthly; 100 for annual.
    one_sided : if True, fit the HP recursively at each point using only
        past data — avoids look-ahead bias for real-time/back-test use.
        Slower; use False for descriptive analysis, True for OOS testing.

    Returns
    -------
    Output gap series in percent of potential (× 100).

    Caveat: the HP filter is well-known to introduce spurious cycles at the
    sample endpoints and to be sensitive to ``lamb`` choice (Hamilton 2018).
    For headline runs, document the choice in the data manifest; consider
    the Hamilton (2018) regression-based alternative as a robustness check.
    """
    try:
        from statsmodels.tsa.filters.hp_filter import hpfilter
    except ImportError as e:
        raise ImportError("HP filter requires statsmodels") from e

    if not one_sided:
        cycle, _ = hpfilter(log_gdp.dropna(), lamb=lamb)
        return (cycle * 100).reindex(log_gdp.index)

    # One-sided: refit at each t using only data up to t. O(T^2) but honest.
    s = log_gdp.dropna()
    out = pd.Series(np.nan, index=s.index)
    for t in range(20, len(s)):  # need a few periods to fit
        cycle_t, _ = hpfilter(s.iloc[: t + 1], lamb=lamb)
        out.iloc[t] = cycle_t.iloc[-1] * 100
    return out.reindex(log_gdp.index)


# =============================================================================
# REER → log change
# =============================================================================

def reer_log_change(reer_level: pd.Series, periods: int = 1) -> pd.Series:
    """
    Δlog(REER) over ``periods`` (default 1-period change).

    BIS REER levels are typically monthly. The canonical regressor is the
    1-month log change — a strengthening currency (REER ↑) raises inflation
    pressure, so γ < 0 means a strong currency disinflates.
    """
    return np.log(reer_level).diff(periods)


# =============================================================================
# Frequency alignment
# =============================================================================

def align_to_monthly(
    series: pd.Series,
    method: Literal["last", "mean", "ffill", "linear"] = "last",
) -> pd.Series:
    """
    Resample a series to monthly end-of-period using the specified method.

    'last' : period-end snapshot (correct for stock variables like policy rate)
    'mean' : period average (correct for flow variables like CPI YoY when
             aggregating from higher-frequency to monthly)
    'ffill': forward-fill from lower frequency (correct for quarterly
             surveys like BER inflation expectations — value persists
             between observations)
    'linear': linear interpolation (less defensible — adds smoothness that
              isn't in the data; use only when explicitly justified)
    """
    if method == "last":
        return series.resample("ME").last()
    elif method == "mean":
        return series.resample("ME").mean()
    elif method == "ffill":
        return series.resample("ME").ffill()
    elif method == "linear":
        # Reindex to monthly grid then interpolate
        monthly_idx = pd.date_range(series.index.min(), series.index.max(), freq="ME")
        return series.reindex(monthly_idx).interpolate(method="linear")
    else:
        raise ValueError(f"Unknown method: {method!r}")


# =============================================================================
# Real-time alignment (release lag)
# =============================================================================

def apply_release_lag(series: pd.Series, lag_periods: int) -> pd.Series:
    """
    Shift a series forward by ``lag_periods`` to reflect data release lags.

    CPI for month M is typically released mid-month M+1. GDP for quarter Q
    is released ~6 weeks after quarter end. For real-time analysis, the
    series available to the MPC at date t is the value labelled t − lag.

    Example: SA CPI release lag ≈ 1 month → apply_release_lag(cpi, 1).
    """
    return series.shift(lag_periods)


# =============================================================================
# Inflation expectations from BEI
# =============================================================================

def bei_breakeven(nominal_yield: pd.Series, real_yield: pd.Series) -> pd.Series:
    """
    Breakeven inflation: nominal yield − real yield (inflation-linked).

    This is the market-implied inflation expectation at the maturity of the
    pair. For SA: R-bond yield vs equivalent-tenor I-bond. For US: nominal
    Treasury vs TIPS. Cleaner than survey expectations (daily frequency,
    market-disciplined) but contains an inflation risk premium that surveys
    do not.

    Returns a series in percent.
    """
    return nominal_yield - real_yield
