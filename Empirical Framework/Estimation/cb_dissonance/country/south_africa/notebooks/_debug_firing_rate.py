"""
Diagnostic for the 89.8% firing rate seen in 02_l1_intra_sa_real.

Two hypotheses to check:
  (H1) Q-too-rigid: fixed_q_diag=1e-3 keeps TVP coefficients near-constant.
       Cal-window dispersion is tiny -> Sigma_inv is huge -> small absolute
       drift looks like enormous Mahalanobis distance.
  (H2) Baseline mismatch: TVP filter (recursive) settles at a different mean
       than the static OLS fit (whole-cal). Then R = TVP - static has non-zero
       mean over the cal window itself, so D is large everywhere.

Strategy:
  1. Fit the pipeline once and dump TVP path, R, D.
  2. Compare static baseline vs TVP-mean-over-cal baseline (after burn-in).
  3. Re-run TVP with ML-estimated Q (no fix), 1e-3, 1e-2, 1e-1 and compare
     firing rates on cal vs test.
  4. Cross-check with the rolling-window estimator (no state-space prior).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from cb_dissonance.src.data.loaders import load_from_csv
from cb_dissonance.src.data.schema import (
    ReactionFunctionSpec, COL_DATE, COL_INFLATION_TARGET,
)
from cb_dissonance.src.level_1.intra_node.reaction_function import fit_static
from cb_dissonance.src.level_1.intra_node.tvp_estimation import fit_tvp
from cb_dissonance.src.level_1.intra_node.rolling_estimation import fit_rolling
from cb_dissonance.src.level_1.intra_node.diagnostics import (
    dissonance_intra_l1, intensity_test,
)


SPEC = ReactionFunctionSpec(
    country="ZAF", spec_type="AM_MK_EM",
    pi_star=4.5, r_star=2.0,
    inflation_horizon=4, output_gap_horizon=0,
    smoothing=True, include_fx=True, include_foreign_rate=True,
    calibration_window=("2005-01-01", "2014-12-31"),
    test_window=("2015-01-01", "2024-12-31"),
    frequency="M", panel="EM",
)


def banner(s: str) -> None:
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


def load_panel() -> pd.DataFrame:
    df = load_from_csv(
        Path("cb_dissonance/data/raw/sa/sa_panel.csv"),
        country="ZAF",
    )
    df[COL_INFLATION_TARGET] = SPEC.pi_star
    return df


def summarise_coef_path(name: str, theta: pd.DataFrame, baseline: pd.Series,
                       cal_start: str, cal_end: str, burn_in: int = 24) -> None:
    print(f"\n[{name}] TVP path summary (after {burn_in}-mo burn-in)")
    t = theta.iloc[burn_in:].copy()
    cal_mask = (t.index >= cal_start) & (t.index <= cal_end)
    cal = t.loc[cal_mask]
    test = t.loc[t.index > cal_end]
    print(f"  cal-window obs = {len(cal)}, test-window obs = {len(test)}")
    print(f"  {'coef':<8}  {'baseline':>10}  {'cal mean':>10}  {'cal std':>10}  "
          f"{'(cal mean - base)':>18}  {'test mean':>10}  {'test std':>10}")
    for c in theta.columns:
        b = float(baseline.get(c, np.nan))
        cm = float(cal[c].mean())
        cs = float(cal[c].std())
        tm = float(test[c].mean())
        ts = float(test[c].std())
        print(f"  {c:<8}  {b:>10.4f}  {cm:>10.4f}  {cs:>10.4f}  "
              f"{cm - b:>18.4f}  {tm:>10.4f}  {ts:>10.4f}")


def firing_rates(d: pd.Series, fires: pd.Series, cal_end: str) -> tuple[float, float]:
    cal_mask = d.index <= cal_end
    return float(fires[cal_mask].mean()), float(fires[~cal_mask].mean())


def run_one_q(df: pd.DataFrame, q_diag, label: str,
              use_static_baseline: bool = True) -> dict:
    """Run a single TVP+dissonance combo and report cal/test firing rates."""
    static = fit_static(df, SPEC, method="OLS")
    initial_state = static.reduced_form_coefficients.reindex(
        ["const"] + SPEC.regressors
    ).fillna(0).values

    tvp_kwargs = dict(initial_state=initial_state)
    if q_diag is not None:
        n_states = len(initial_state)
        tvp_kwargs["fixed_q_diag"] = np.array([q_diag] * n_states)

    tvp = fit_tvp(df, SPEC, **tvp_kwargs)

    # Choose baseline
    if use_static_baseline:
        baseline = static.structural_coefficients
    else:
        # TVP-mean-over-cal-window (after 24-mo burn-in) as baseline
        burn = 24
        cal_start, cal_end = SPEC.calibration_window
        cal_mask = ((tvp.filtered_structural.index >= cal_start)
                    & (tvp.filtered_structural.index <= cal_end))
        post_burn = tvp.filtered_structural.iloc[burn:]
        cal_post_burn = post_burn.loc[
            (post_burn.index >= cal_start) & (post_burn.index <= cal_end)
        ]
        baseline = cal_post_burn.mean()

    diss = dissonance_intra_l1(
        coefficients_over_time=tvp.filtered_structural,
        baseline=baseline,
        spec=SPEC,
        method_label=label,
    )
    cal_fire, test_fire = firing_rates(
        diss.d_mahalanobis, diss.fires, SPEC.calibration_window[1]
    )
    return {
        "label": label,
        "q_diag": q_diag,
        "baseline": "static" if use_static_baseline else "TVP-cal-mean",
        "tau": diss.threshold,
        "cal_fire_rate": cal_fire,
        "test_fire_rate": test_fire,
        "tvp": tvp,
        "static": static,
        "baseline_vec": baseline,
        "diss": diss,
    }


def main():
    df = load_panel()
    banner("LOADED PANEL")
    print(df.describe().round(3).to_string())

    # ---------- (1) Headline run (matches 02_l1_intra_sa_real.py) ----------
    banner("(1) HEADLINE: fixed Q=1e-3, baseline=static OLS")
    head = run_one_q(df, q_diag=1e-3, label="headline",
                     use_static_baseline=True)
    print(f"  tau = {head['tau']:.4f}")
    print(f"  cal firing  = {head['cal_fire_rate']:.1%}")
    print(f"  test firing = {head['test_fire_rate']:.1%}")
    summarise_coef_path(
        "TVP filtered", head["tvp"].filtered_structural, head["baseline_vec"],
        *SPEC.calibration_window,
    )

    # ---------- (2) Try TVP-mean-cal baseline ----------
    banner("(2) Same Q=1e-3, baseline=TVP mean over cal (post burn-in)")
    alt = run_one_q(df, q_diag=1e-3, label="tvp-cal-mean-baseline",
                    use_static_baseline=False)
    print(f"  tau = {alt['tau']:.4f}")
    print(f"  cal firing  = {alt['cal_fire_rate']:.1%}")
    print(f"  test firing = {alt['test_fire_rate']:.1%}")

    # ---------- (3) Sweep Q with static baseline ----------
    banner("(3) Q SWEEP (static baseline)")
    print(f"  {'label':<22}  {'Q':>10}  {'tau':>10}  {'cal_fire':>10}  {'test_fire':>10}")
    sweep = []
    for q in [None, 1e-4, 1e-3, 1e-2, 1e-1]:
        label = f"Q={'ML' if q is None else q}"
        try:
            r = run_one_q(df, q_diag=q, label=label, use_static_baseline=True)
            print(f"  {label:<22}  {str(q):>10}  {r['tau']:>10.4f}  "
                  f"{r['cal_fire_rate']:>10.1%}  {r['test_fire_rate']:>10.1%}")
            sweep.append(r)
        except Exception as e:
            print(f"  {label:<22}  failed: {e}")

    # ---------- (4) Sweep Q with TVP-cal-mean baseline ----------
    banner("(4) Q SWEEP (TVP-cal-mean baseline)")
    print(f"  {'label':<22}  {'Q':>10}  {'tau':>10}  {'cal_fire':>10}  {'test_fire':>10}")
    for q in [None, 1e-4, 1e-3, 1e-2, 1e-1]:
        label = f"Q={'ML' if q is None else q}"
        try:
            r = run_one_q(df, q_diag=q, label=label, use_static_baseline=False)
            print(f"  {label:<22}  {str(q):>10}  {r['tau']:>10.4f}  "
                  f"{r['cal_fire_rate']:>10.1%}  {r['test_fire_rate']:>10.1%}")
        except Exception as e:
            print(f"  {label:<22}  failed: {e}")

    # ---------- (5) Rolling-window comparator ----------
    banner("(5) ROLLING-WINDOW COMPARATOR (window=60m)")
    rolling = fit_rolling(df, SPEC, window=60)
    rc = rolling.structural_coefficients
    print(f"  Windows fit: {len(rc)}")
    print(f"  {'coef':<8}  {'cal mean':>10}  {'cal std':>10}  {'min':>10}  {'max':>10}")
    cal_end_ts = pd.Timestamp(SPEC.calibration_window[1])
    cal_mask = rc.index <= cal_end_ts
    for c in rc.columns:
        cal_vals = rc.loc[cal_mask, c]
        full_vals = rc[c]
        print(f"  {c:<8}  {cal_vals.mean():>10.4f}  {cal_vals.std():>10.4f}  "
              f"{full_vals.min():>10.4f}  {full_vals.max():>10.4f}")

    # ---------- (6) Look at D series for the headline run ----------
    banner("(6) HEADLINE D-trace at key dates")
    d = head["diss"].d_mahalanobis
    sel = ["2007-01-31", "2008-12-31", "2010-12-31", "2014-12-31",
           "2016-12-31", "2018-12-31", "2020-12-31", "2022-12-31", "2024-12-31"]
    for s in sel:
        ts = pd.Timestamp(s)
        if ts in d.index:
            v = d.loc[ts]
            print(f"  {s}  D = {v:.3f}   D/tau = {v/head['tau']:.2f}x")


if __name__ == "__main__":
    main()
