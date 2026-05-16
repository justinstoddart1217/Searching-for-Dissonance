from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# HERE = .../country/south_africa/src/nodes/level_1/vertical
COUNTRY_ROOT = HERE.parents[3]              # .../country/south_africa
PACKAGE_ROOT = COUNTRY_ROOT.parent.parent   # .../cb_dissonance (contains country/)

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from country.south_africa.src.data.loaders import build_sarb_l1_dataset
from country.south_africa.src.nodes.level_1.reaction_function import SARBReactionFunction
from country.south_africa.src.nodes.level_1.vertical.l1_l3a import correlation_test, consistency, dissonance


def _status(ok: bool) -> str:
    return 'PASS' if ok else 'FAIL'


def main() -> int:
    raw_dir = COUNTRY_ROOT / 'data' / 'raw'
    processed_dir = COUNTRY_ROOT / 'data' / 'processed'

    print('=' * 72)
    print('SARB L1 -> L3a Edge Test')
    print('=' * 72)

    print()
    print('[1] Build dataset')
    print('-' * 72)
    ds = build_sarb_l1_dataset(raw_dir, processed_dir, write_parquet=True)
    print(f"  {ds.metadata['n_obs']} obs ({ds.metadata['sample_start']} -> {ds.metadata['sample_end']})")

    print()
    print('[2] Fit reaction function (OLS+HAC)')
    print('-' * 72)
    fit = SARBReactionFunction().fit(ds.data)
    d = fit.diagnostics
    print(f"  rho            = {fit.theta_hat['rho']:+.4f}")
    print(f"  R^2 calib      = {d['r_squared']:.4f}")
    print(f"  sigma_ratio    = {d['sigma_ratio']:.4f}  (calib)")
    print(f"  sigma(eps)     = {d['sigma_eps']:.4f}  (calib)")
    print(f"  N calib        = {d['n_obs_calibration']}")
    print(f"  N oos          = {d['n_obs_oos']}")

    print()
    print('[3] CORRELATION TEST (gate)')
    print('-' * 72)
    ct = correlation_test(fit)
    print('  Pre-registered admissibility bars (calibration only):')
    print(f"    R^2  >= {ct.metadata['pre_reg_r2_bar']:.2f}    -> R^2  calib = {ct.r2_calib:.4f}  [{_status(ct.passes_r2_calib)}]")
    print(f"    s_r  <= {ct.metadata['pre_reg_sigma_ratio_bar']:.2f}    -> s_r  calib = {ct.sigma_ratio_calib:.4f}  [{_status(ct.passes_sigma_ratio_calib)}]")
    print()
    print('  Out-of-sample (diagnostic, not a gate):')
    print(f"    R^2  oos       = {ct.r2_oos:.4f}  [{_status(ct.passes_r2_oos)}]")
    print(f"    s_r  oos       = {ct.sigma_ratio_oos:.4f}  [{_status(ct.passes_sigma_ratio_oos)}]")
    print()
    print(f"  EDGE STATUS: {'ADMISSIBLE' if ct.admissible else 'REJECTED'}")

    if not ct.admissible:
        print()
        print('  Edge rejected by gate test. Halting before computing R/D.')
        return 1

    print()
    print('[4] CONSISTENCY  R(t) = i_t - i*_t')
    print('-' * 72)
    cn = consistency(fit)
    md = cn.metadata
    print(f"  Sample mean R   = {md['sample_mean']:+.4f}    (E[R]=0 in equilibrium)")
    print(f"  Sample SD R     = {md['sample_std_dev']:.4f}")
    print(f"  N obs           = {md['n_obs']}  ({md['index_start']} -> {md['index_end']})")

    print()
    print('[5] DISSONANCE  D(t) = |R(t)|,  tau = 2 * sigma(eps_calib)')
    print('-' * 72)
    di = dissonance(fit)
    md = di.metadata
    print(f"  sigma(eps_calib)= {md['sigma_eps_calib']:.4f}")
    print(f"  threshold tau   = {md['tau']:.4f}")
    print()
    print(f"  Flagged total   = {md['n_flagged_total']:>3}   ({100*md['flagged_fraction_total']:.1f}%)")
    print(f"    in calib      = {md['n_flagged_calib']:>3}   ({100*md['flagged_fraction_calib']:.1f}%)")
    print(f"    in oos        = {md['n_flagged_oos']:>3}   ({100*md['flagged_fraction_oos']:.1f}%)")

    print()
    print('  FLAGGED MONTHS (signed R(t)):')
    print('  ' + '-' * 68)
    flagged_dates = di.D.index[di.flagged]
    if len(flagged_dates) == 0:
        print('    (none)')
    else:
        import pandas as pd
        calib_end_ts = pd.Timestamp(fit.metadata['calibration_window'][1]) + pd.offsets.MonthEnd(0)
        last_year = None
        for date in flagged_dates:
            year = date.year
            month = date.strftime('%b')
            signed = float(fit.residuals.loc[date])
            d_val = float(di.D.loc[date])
            window = 'calib' if date <= calib_end_ts else 'oos'
            marker = '  ' if year == last_year else f'  {year}'
            print(f'    {marker:<8} {month}   R = {signed:+.3f}    D = {d_val:.3f}    ({window})')
            last_year = year

    print()
    print('=' * 72)
    print('Done.')
    print('=' * 72)
    return 0


if __name__ == '__main__':
    sys.exit(main())
