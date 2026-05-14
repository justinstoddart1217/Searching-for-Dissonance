"""
Loaders for real reaction-function inputs.

The contract: every loader returns a DataFrame conforming to ``schema.py``
canonical columns. Sourcing, frequency alignment, and proxy choices are
handled here so estimation code stays clean.

Three common workflows:

  1. Pre-aligned monthly CSV (Bloomberg/Refinitiv export, one wide file):
         df = load_from_csv("data/raw/sa/sa_panel.csv", country="ZAF")

  2. Series-by-series assembly (each variable from a different source):
         df = load_from_series_dict({
             COL_POLICY_RATE: repo_rate_series,
             COL_INFLATION: cpi_series,
             ...
         }, country="ZAF")

  3. Mixed-frequency: pass quarterly series and resample inside the loader:
         df = load_from_series_dict(
             {COL_OUTPUT_GAP: quarterly_gap, COL_POLICY_RATE: monthly_repo, ...},
             country="ZAF", target_frequency="M",
         )

The data manifest (config/data_manifest_<country>.yaml) documents the
source and transformation for each canonical column — this is the data
appendix in machine-readable form.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Union

import pandas as pd

from cb_dissonance.src.data.schema import (
    COL_DATE,
    COL_COUNTRY,
    COL_POLICY_RATE,
    COL_INFLATION,
    COL_INFLATION_TARGET,
    COL_OUTPUT_GAP,
    COL_INFLATION_EXP_H,
    COL_REER_CHANGE,
    COL_FOREIGN_RATE,
    COL_COMMODITY_INDEX,
)


# Canonical columns the loader knows how to populate.
_CANONICAL_COLUMNS = [
    COL_POLICY_RATE,
    COL_INFLATION,
    COL_INFLATION_TARGET,
    COL_OUTPUT_GAP,
    COL_INFLATION_EXP_H,
    COL_REER_CHANGE,
    COL_FOREIGN_RATE,
    COL_COMMODITY_INDEX,
]


# =============================================================================
# CSV loader (wide format — one column per canonical series)
# =============================================================================

def load_from_csv(
    path: Union[str, Path],
    country: str,
    column_map: Optional[Mapping[str, str]] = None,
    date_column: str = "date",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Load a wide-format CSV where each column is one canonical series.

    Parameters
    ----------
    path : path to a CSV. Required column: ``date``. Other columns should
        either be already named after the canonical schema, or be remapped
        via ``column_map``.
    country : ISO-3 (added as a column).
    column_map : optional mapping from CSV column names → canonical names.
        Example: {"SARRRD_Index": "policy_rate", "SACPYOY_Index": "inflation"}.
        Columns not in the map and not already canonical are dropped with a
        warning so estimation code does not silently pick up junk.
    date_column : name of the date column in the CSV.

    Returns
    -------
    DataFrame matching ``schema.py``: sorted by date, one row per period,
    canonical column names.
    """
    df = pd.read_csv(path, parse_dates=[date_column] if parse_dates else None)

    if date_column != COL_DATE:
        df = df.rename(columns={date_column: COL_DATE})

    if column_map:
        df = df.rename(columns=dict(column_map))

    # Keep only date + canonical columns; drop the rest with a notice
    keep = [COL_DATE] + [c for c in _CANONICAL_COLUMNS if c in df.columns]
    dropped = [c for c in df.columns if c not in keep and c != COL_DATE]
    if dropped:
        import warnings
        warnings.warn(
            f"load_from_csv: dropping non-canonical columns {dropped}. "
            f"Add them to column_map if they should be renamed.",
            stacklevel=2,
        )
    df = df[keep].copy()
    df[COL_COUNTRY] = country
    df = df.sort_values(COL_DATE).reset_index(drop=True)
    return df


# =============================================================================
# Series-dict loader (each variable from its own pd.Series)
# =============================================================================

def load_from_series_dict(
    series: Mapping[str, pd.Series],
    country: str,
    target_frequency: str = "M",
    quarterly_ffill: bool = True,
) -> pd.DataFrame:
    """
    Assemble a panel DataFrame from a dict of {canonical_column: pd.Series}.

    Each series must have a DatetimeIndex. Mixed frequencies are aligned to
    ``target_frequency`` (default monthly). Quarterly series (e.g. output gap
    from quarterly GDP, BER inflation expectations) are forward-filled to the
    target frequency when ``quarterly_ffill=True``.

    This is the typical real-data workflow when each variable comes from a
    different source: Bloomberg for the policy rate, StatsSA for CPI, BER
    for inflation expectations, BIS for REER, FRED for the foreign rate.
    """
    if target_frequency not in ("M", "Q"):
        raise ValueError(f"target_frequency must be 'M' or 'Q', got {target_frequency!r}")

    target_freq_alias = "ME" if target_frequency == "M" else "QE"

    aligned = {}
    for col, s in series.items():
        if not isinstance(s.index, pd.DatetimeIndex):
            raise TypeError(f"{col}: index must be DatetimeIndex (got {type(s.index)})")

        # Detect source frequency heuristically
        inferred = pd.infer_freq(s.index)
        src_is_quarterly = inferred is not None and inferred.startswith("Q")

        if src_is_quarterly and target_frequency == "M":
            if not quarterly_ffill:
                raise ValueError(
                    f"{col}: quarterly source but quarterly_ffill=False — "
                    f"set =True to forward-fill into monthly, or pre-align before passing."
                )
            # Resample to monthly end, forward-fill within quarter
            s = s.resample("ME").ffill()
        else:
            s = s.resample(target_freq_alias).last()

        aligned[col] = s

    # Outer-join on the common monthly grid
    df = pd.concat(aligned, axis=1)
    df.index.name = COL_DATE
    df = df.reset_index()
    df[COL_COUNTRY] = country
    df = df.sort_values(COL_DATE).reset_index(drop=True)
    return df


# =============================================================================
# Template writer
# =============================================================================

def write_template_csv(
    path: Union[str, Path],
    start: str = "2005-01-31",
    end: str = "2024-12-31",
    frequency: str = "M",
    panel: str = "EM",
) -> None:
    """
    Write an empty CSV with the canonical schema column headers, dated
    monthly across the requested range. Open in Excel, paste in real values
    from Bloomberg/Refinitiv/SARB by column, save as CSV.

    EM panels need REER and foreign rate; DM panels can leave those blank.
    """
    freq_alias = "ME" if frequency == "M" else "QE"
    dates = pd.date_range(start, end, freq=freq_alias)
    df = pd.DataFrame({COL_DATE: dates})

    base_cols = [
        COL_POLICY_RATE,
        COL_INFLATION,
        COL_INFLATION_TARGET,
        COL_OUTPUT_GAP,
        COL_INFLATION_EXP_H,
    ]
    em_cols = [COL_REER_CHANGE, COL_FOREIGN_RATE]
    cols = base_cols + (em_cols if panel == "EM" else [])
    for c in cols:
        df[c] = pd.NA

    df.to_csv(path, index=False)
    print(f"Wrote template with {len(dates)} rows and columns: {[COL_DATE] + cols}")
    print(f"  → {path}")
