from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COUNTRY_ROOT = HERE.parent
PACKAGE_ROOT = COUNTRY_ROOT.parent.parent

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from country.south_africa.src.data.loaders import build_sarb_l1_dataset
from country.south_africa.src.nodes.level_1.reaction_function import SARBReactionFunction


PRE_REG_R2_BAR = 0.84
PRE_REG_SIGMA_RATIO_BAR = 0.40
AM_REFERENCE = {'alpha': 1.5, 'beta': 0.5, 'gamma': None, 'delta': None, 'rho': 0.85}


def _status(value: float, threshold: float, direction: str) -> str:
    if direction == '>=':
        return 'PASS' if value >= threshold else 'FAIL'
    if direction == '<=':
        return 'PASS' if value <= threshold else 'FAIL'
    return '???'


def main() -> int:
    raw_dir = COUNTRY_ROOT / 'data' / 'raw'
    processed_dir = COUNTRY_ROOT / 'data' / 'processed'

    print('=' * 72)
    print('SARB L1 Reaction Function | Diagnostics Run')
    print('=' * 72)

    print()
    print('[1] Build monthly dataset')
    print('-' * 72)
    ds = build_sarb_l1_dataset(raw_dir, processed_dir, write_parquet=True)
    md = ds.metadata
    print(f"  Sample        : {md['sample_start']} -> {md['sample_end']}  ({md['n_obs']} obs)")
    print(f"  Target        : {md['target']}")
    print(f"  Regressors    : {md['regressors']}")
    print(f"  Controls      : {md['exogenous_controls']}")
    print(f"  HP lambda     : {md['hp_lambda_quarterly']} (quarterly, one-sided real-time)")
    print(f"  Expectations  : {md['expectations_horizon']}")
    print(f"  Parquet at    : {processed_dir / 'sarb_l1_dataset_v1.parquet'}")

    print()
    print('[2] Fit reaction function (OLS + Newey-West HAC, maxlags=6)')
    print('-' * 72)
    fit = SARBReactionFunction().fit(ds.data)
    d = fit.diagnostics

    r2 = d['r_squared']
    ratio_c = d['sigma_ratio']
    ratio_o = d['sigma_ratio_oos']

    print(f"  Window calib  : {fit.metadata['calibration_window']}  N={d['n_obs_calibration']}")
    print(f"  Window oos    : {fit.metadata['oos_window']}  N={d['n_obs_oos']}")
    print()
    print('  PRE-REGISTERED ADMISSIBILITY BARS')
    print(f"    R^2 >= {PRE_REG_R2_BAR}")
    print(f"      calib R^2          = {r2:.4f}    [{_status(r2, PRE_REG_R2_BAR, '>=')}]")
    print(f"    sigma_eps/sigma_i <= {PRE_REG_SIGMA_RATIO_BAR}")
    print(f"      calib sigma_ratio  = {ratio_c:.4f}    [{_status(ratio_c, PRE_REG_SIGMA_RATIO_BAR, '<=')}]")
    print(f"      oos   sigma_ratio  = {ratio_o:.4f}    [{_status(ratio_o, PRE_REG_SIGMA_RATIO_BAR, '<=')}]")
    print()
    print('  ADDITIONAL DIAGNOSTICS')
    print(f"    R^2 adj              = {d['adj_r_squared']:.4f}")
    print(f"    Durbin-Watson        = {d['durbin_watson']:.4f}  (~2 = no resid AC)")
    print(f"    sigma(eps) calib     = {d['sigma_eps']:.4f}")
    print(f"    sigma(i_t)  calib    = {d['sigma_i']:.4f}")
    print(f"    sigma(eps) oos       = {d['sigma_eps_oos']:.4f}")
    print(f"    sigma(i_t)  oos      = {d['sigma_i_oos']:.4f}")
    print(f"    missing oos fraction = {d['missing_oos_fraction']:.4f}")
    print(f"    structural_reliable  = {d['structural_reliable']}")

    print()
    print('[3] Estimated parameters (HAC standard errors)')
    print('-' * 72)
    print(f"  rho      = {fit.theta_hat['rho']:+.4f}  (SE {fit.theta_se['rho_se']:.4f})   [smoothing; A-M ref ~{AM_REFERENCE['rho']}]")
    print()
    if d['structural_reliable']:
        print('  Structural coefficients (delta-method SEs):')
        for k in ['alpha', 'beta', 'gamma', 'delta']:
            val = fit.theta_hat[k]
            se = fit.theta_se[f'{k}_se']
            ref = AM_REFERENCE[k]
            ref_str = f"A-M ref ~{ref}" if ref is not None else "A-M ref n/a"
            print(f"    {k:<8} = {val:+.4f}  (SE {se:.4f})   [{ref_str}]")
    else:
        print('  Structural recovery suppressed (rho >= 0.95 guard).')
        print('  Reduced-form linear coefficients:')
        for k in ['c', 'rho', 'theta_1', 'theta_2', 'theta_3', 'theta_4']:
            v = fit.theta_linear[k]
            se = fit.theta_linear_se.get(f'{k}_se', float('nan'))
            print(f"    {k:<8} = {v:+.4f}  (SE {se:.4f})")
        print()
        print('  Mapping: theta_1=pi_gap, theta_2=output_gap, theta_3=delta_reer, theta_4=fed_funds')
        print('  To recover structural: alpha = theta_1 / (1 - rho), etc.')

    print()
    print('=' * 72)
    print('Done.')
    print('=' * 72)
    return 0


if __name__ == '__main__':
    sys.exit(main())
