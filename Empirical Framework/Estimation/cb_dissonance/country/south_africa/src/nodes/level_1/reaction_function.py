from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True, eq=False)
class SARBReactionFunctionFit:
    theta_hat: dict
    theta_se: dict
    theta_linear: dict
    theta_linear_se: dict
    i_star: pd.Series
    residuals: pd.Series
    diagnostics: dict
    metadata: dict


class SARBReactionFunction:

    PI_TARGET: float = 4.5

    GFC_START: str = '2008-10'
    GFC_END:   str = '2009-09'

    DEFAULT_CALIBRATION_WINDOW: tuple[str, str] = ('2005-01', '2014-12')
    DEFAULT_OOS_WINDOW:         tuple[str, str] = ('2015-01', '2024-12')

    REGRESSORS: tuple[str, ...] = (
        'i_lag1',
        'pi_gap',
        'output_gap_nowcast',
        'delta_zar_reer',
        'fed_funds',
    )

    LINEAR_TO_COL: dict[str, str] = {
        'theta_1': 'pi_gap',
        'theta_2': 'output_gap_nowcast',
        'theta_3': 'delta_zar_reer',
        'theta_4': 'fed_funds',
    }

    LINEAR_TO_STRUCT: dict[str, str] = {
        'theta_1': 'alpha',
        'theta_2': 'beta',
        'theta_3': 'gamma',
        'theta_4': 'delta',
    }

    HAC_MAXLAGS: int = 6

    RHO_GUARD_THRESHOLD: float = 0.95

    @property
    def linear_param_names(self) -> list[str]:
        return ['theta_1', 'theta_2', 'theta_3', 'theta_4']

    def fit(
        self,
        data: pd.DataFrame,
        calibration_window: tuple[str, str] | None = None,
        oos_window: tuple[str, str] | None = None,
        gfc_dummy: bool = True,
        method: str = 'ols',
    ) -> SARBReactionFunctionFit:
        if method != 'ols':
            raise NotImplementedError(
                f"Method '{method}' not yet supported. Use 'ols'."
            )

        if calibration_window is None:
            calibration_window = self.DEFAULT_CALIBRATION_WINDOW
        if oos_window is None:
            oos_window = self.DEFAULT_OOS_WINDOW

        y_calib, X_calib = self._prepare_data(
            data, calibration_window, gfc_dummy=gfc_dummy
        )

        ols_result = sm.OLS(y_calib, X_calib).fit(
            cov_type='HAC',
            cov_kwds={'maxlags': self.HAC_MAXLAGS},
        )

        theta_linear, theta_linear_se = self._extract_linear_params(ols_result)
        theta_hat, theta_se, structural_reliable = self._recover_structural(
            theta_linear, ols_result
        )

        n_oos_raw = len(data.loc[oos_window[0]:oos_window[1]])
        y_oos, X_oos = self._prepare_data(
            data, oos_window, gfc_dummy=gfc_dummy, min_obs=0
        )
        i_star_oos = ols_result.predict(X_oos)
        residuals_oos = y_oos - i_star_oos

        i_star_calib = ols_result.fittedvalues
        residuals_calib = y_calib - i_star_calib

        i_star = pd.concat([i_star_calib, i_star_oos]).sort_index()
        i_star.name = 'i_star'
        residuals = pd.concat([residuals_calib, residuals_oos]).sort_index()
        residuals.name = 'residuals'

        n_oos_kept = len(y_oos)
        missing_oos_fraction = (
            (n_oos_raw - n_oos_kept) / n_oos_raw if n_oos_raw > 0 else 0.0
        )

        sigma_eps_calib = float(residuals_calib.std(ddof=1))
        sigma_i_calib   = float(y_calib.std(ddof=1))
        sigma_eps_oos   = float(residuals_oos.std(ddof=1)) if len(residuals_oos) > 1 else float('nan')
        sigma_i_oos     = float(y_oos.std(ddof=1)) if len(y_oos) > 1 else float('nan')

        diagnostics = {
            'r_squared':            float(ols_result.rsquared),
            'adj_r_squared':        float(ols_result.rsquared_adj),
            'durbin_watson':        float(sm.stats.stattools.durbin_watson(ols_result.resid)),
            'n_obs_calibration':    int(ols_result.nobs),
            'n_obs_oos':            n_oos_kept,
            'missing_oos_fraction': missing_oos_fraction,
            'structural_reliable':  structural_reliable,
            'sigma_eps':            sigma_eps_calib,
            'sigma_i':              sigma_i_calib,
            'sigma_ratio':          sigma_eps_calib / sigma_i_calib if sigma_i_calib > 0 else float('nan'),
            'sigma_eps_oos':        sigma_eps_oos,
            'sigma_i_oos':          sigma_i_oos,
            'sigma_ratio_oos':      (sigma_eps_oos / sigma_i_oos) if (sigma_i_oos and sigma_i_oos > 0) else float('nan'),
        }

        metadata = {
            'method':              method,
            'calibration_window':  tuple(calibration_window),
            'oos_window':          tuple(oos_window),
            'gfc_dummy_used':      bool(gfc_dummy and 'gfc_dummy' in data.columns),
            'pi_target':           self.PI_TARGET,
            'specification':       'Aron-Muellbauer (2007) CGG forward-smoothed',
        }

        return SARBReactionFunctionFit(
            theta_hat=theta_hat,
            theta_se=theta_se,
            theta_linear=theta_linear,
            theta_linear_se=theta_linear_se,
            i_star=i_star,
            residuals=residuals,
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def _prepare_data(
        self,
        data: pd.DataFrame,
        window: tuple[str, str],
        *,
        gfc_dummy: bool = True,
        min_obs: int = 30,
    ) -> tuple[pd.Series, pd.DataFrame]:
        required = ['i_t'] + list(self.REGRESSORS)
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for SARB: {missing}. "
                f"Expected schema: {required}."
            )

        start, end = window
        sliced = data.loc[start:end]
        if sliced.empty:
            raise ValueError(
                f"Window {window} produced empty slice. "
                f"Check the date index of ``data``."
            )

        y = sliced['i_t']
        X = sliced[list(self.REGRESSORS)].copy()

        if gfc_dummy and 'gfc_dummy' in sliced.columns:
            X['gfc_dummy'] = sliced['gfc_dummy']

        X = sm.add_constant(X, has_constant='add')

        combined = pd.concat([y, X], axis=1).dropna()
        y_clean = combined['i_t']
        X_clean = combined.drop(columns='i_t')

        n_obs = len(y_clean)
        if n_obs < min_obs:
            raise ValueError(
                f"Insufficient observations after NaN-dropping: {n_obs} "
                f"(need >= {min_obs}). Window={window}, SARB."
            )

        return y_clean, X_clean

    def _extract_linear_params(self, ols_result) -> tuple[dict, dict]:
        params = ols_result.params
        bse = ols_result.bse

        theta_linear = {
            'c':   float(params['const']),
            'rho': float(params['i_lag1']),
        }
        theta_linear_se = {
            'c_se':   float(bse['const']),
            'rho_se': float(bse['i_lag1']),
        }

        for lin_name in self.linear_param_names:
            col = self.LINEAR_TO_COL[lin_name]
            theta_linear[lin_name] = float(params[col])
            theta_linear_se[f'{lin_name}_se'] = float(bse[col])

        return theta_linear, theta_linear_se

    def _recover_structural(
        self,
        theta_linear: dict,
        ols_result,
    ) -> tuple[dict, dict, bool]:
        rho_hat = theta_linear['rho']
        cov = ols_result.cov_params()

        theta_hat: dict = {'rho': rho_hat}
        theta_se: dict = {'rho_se': float(np.sqrt(cov.loc['i_lag1', 'i_lag1']))}

        if rho_hat >= self.RHO_GUARD_THRESHOLD:
            for lin_name in self.linear_param_names:
                struct_name = self.LINEAR_TO_STRUCT[lin_name]
                theta_hat[struct_name] = float('nan')
                theta_se[f'{struct_name}_se'] = float('nan')
            return theta_hat, theta_se, False

        one_minus_rho = 1.0 - rho_hat
        var_rho = cov.loc['i_lag1', 'i_lag1']

        for lin_name in self.linear_param_names:
            struct_name = self.LINEAR_TO_STRUCT[lin_name]
            col = self.LINEAR_TO_COL[lin_name]
            theta_j = theta_linear[lin_name]

            theta_hat[struct_name] = theta_j / one_minus_rho

            d_theta = 1.0 / one_minus_rho
            d_rho = theta_j / (one_minus_rho ** 2)
            var_theta = cov.loc[col, col]
            cov_theta_rho = cov.loc[col, 'i_lag1']

            var_struct = (
                d_theta ** 2 * var_theta
                + d_rho ** 2 * var_rho
                + 2.0 * d_theta * d_rho * cov_theta_rho
            )
            theta_se[f'{struct_name}_se'] = (
                float(np.sqrt(var_struct)) if var_struct >= 0 else float('nan')
            )

        return theta_hat, theta_se, True
