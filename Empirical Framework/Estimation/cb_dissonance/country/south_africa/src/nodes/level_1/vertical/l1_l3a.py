from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


EDGE_ID: str = 'L1-L3a'

PRE_REG_R2_BAR: float = 0.84
PRE_REG_SIGMA_RATIO_BAR: float = 0.40
DEFAULT_THRESHOLD_MULTIPLIER: float = 2.0


@dataclass(frozen=True, eq=False)
class CorrelationTestResult:
    r2_calib: float
    r2_oos: float
    sigma_ratio_calib: float
    sigma_ratio_oos: float
    passes_r2_calib: bool
    passes_sigma_ratio_calib: bool
    passes_r2_oos: bool
    passes_sigma_ratio_oos: bool
    admissible: bool
    metadata: dict


@dataclass(frozen=True, eq=False)
class ConsistencyResult:
    R: pd.Series
    metadata: dict


@dataclass(frozen=True, eq=False)
class DissonanceResult:
    D: pd.Series
    threshold_tau: float
    flagged: pd.Series
    metadata: dict


def correlation_test(fit) -> CorrelationTestResult:
    calib_start, calib_end = fit.metadata['calibration_window']
    oos_start, oos_end = fit.metadata['oos_window']

    resid_calib = fit.residuals.loc[calib_start:calib_end]
    resid_oos = fit.residuals.loc[oos_start:oos_end]
    i_star_calib = fit.i_star.loc[calib_start:calib_end]
    i_star_oos = fit.i_star.loc[oos_start:oos_end]
    i_t_calib = i_star_calib + resid_calib
    i_t_oos = i_star_oos + resid_oos

    r2_c = _bivariate_r2(i_t_calib, resid_calib)
    r2_o = _bivariate_r2(i_t_oos, resid_oos)
    sr_c = _sigma_ratio(i_t_calib, resid_calib)
    sr_o = _sigma_ratio(i_t_oos, resid_oos)

    passes_r2_c = r2_c >= PRE_REG_R2_BAR
    passes_sr_c = sr_c <= PRE_REG_SIGMA_RATIO_BAR
    passes_r2_o = r2_o >= PRE_REG_R2_BAR
    passes_sr_o = sr_o <= PRE_REG_SIGMA_RATIO_BAR
    admissible = passes_r2_c and passes_sr_c

    metadata = {
        'edge': EDGE_ID,
        'gate_definition': 'Bivariate R²(i_t, i*_t) and sigma(eps)/sigma(i_t) on calibration window',
        'pre_reg_r2_bar': PRE_REG_R2_BAR,
        'pre_reg_sigma_ratio_bar': PRE_REG_SIGMA_RATIO_BAR,
        'calibration_window': (calib_start, calib_end),
        'oos_window': (oos_start, oos_end),
        'n_obs_calib': int(len(i_t_calib)),
        'n_obs_oos': int(len(i_t_oos)),
        'admissibility_rule': 'admissible = passes_r2_calib AND passes_sigma_ratio_calib',
    }

    return CorrelationTestResult(
        r2_calib=r2_c,
        r2_oos=r2_o,
        sigma_ratio_calib=sr_c,
        sigma_ratio_oos=sr_o,
        passes_r2_calib=passes_r2_c,
        passes_sigma_ratio_calib=passes_sr_c,
        passes_r2_oos=passes_r2_o,
        passes_sigma_ratio_oos=passes_sr_o,
        admissible=admissible,
        metadata=metadata,
    )


def consistency(fit) -> ConsistencyResult:
    R = fit.residuals.copy()
    R.name = f'R_{EDGE_ID.lower().replace("-", "_")}'

    metadata = {
        'edge': EDGE_ID,
        'definition': 'R(t) = i_t - i*_t  (signed residuals from reaction function)',
        'expected_equilibrium': 'E[R] = 0',
        'sample_mean': float(R.mean()),
        'sample_std_dev': float(R.std(ddof=1)),
        'n_obs': int(len(R)),
        'index_start': str(R.index.min().date()),
        'index_end': str(R.index.max().date()),
    }

    return ConsistencyResult(R=R, metadata=metadata)


def dissonance(
    fit,
    threshold_multiplier: float = DEFAULT_THRESHOLD_MULTIPLIER,
) -> DissonanceResult:
    calib_start, calib_end = fit.metadata['calibration_window']
    oos_start, oos_end = fit.metadata['oos_window']

    sigma_eps_calib = float(fit.residuals.loc[calib_start:calib_end].std(ddof=1))
    tau = threshold_multiplier * sigma_eps_calib

    D = fit.residuals.abs()
    D.name = f'D_{EDGE_ID.lower().replace("-", "_")}'
    flagged = D > tau
    flagged.name = f'flag_{EDGE_ID.lower().replace("-", "_")}'

    flagged_oos = flagged.loc[oos_start:oos_end]
    flagged_calib = flagged.loc[calib_start:calib_end]

    metadata = {
        'edge': EDGE_ID,
        'definition': 'D(t) = |R(t)|',
        'threshold_rule': f'tau = {threshold_multiplier} * sigma(eps_calib)',
        'sigma_eps_calib': sigma_eps_calib,
        'tau': tau,
        'threshold_multiplier': threshold_multiplier,
        'n_flagged_total': int(flagged.sum()),
        'n_flagged_calib': int(flagged_calib.sum()),
        'n_flagged_oos': int(flagged_oos.sum()),
        'flagged_fraction_total': float(flagged.mean()),
        'flagged_fraction_calib': float(flagged_calib.mean()),
        'flagged_fraction_oos': float(flagged_oos.mean()),
    }

    return DissonanceResult(
        D=D,
        threshold_tau=tau,
        flagged=flagged,
        metadata=metadata,
    )


def _bivariate_r2(y: pd.Series, eps: pd.Series) -> float:
    if len(y) < 2:
        return float('nan')
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 0:
        return float('nan')
    ssr = float((eps ** 2).sum())
    return 1.0 - ssr / sst


def _sigma_ratio(y: pd.Series, eps: pd.Series) -> float:
    if len(y) < 2:
        return float('nan')
    sigma_y = float(y.std(ddof=1))
    if sigma_y <= 0:
        return float('nan')
    sigma_eps = float(eps.std(ddof=1))
    return sigma_eps / sigma_y
