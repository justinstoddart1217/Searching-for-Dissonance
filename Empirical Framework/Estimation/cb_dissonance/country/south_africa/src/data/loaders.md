# `loaders.py` — SARB Level 1 Data Pipeline

Reads the six raw input CSVs from `country/south_africa/data/raw/`, harmonises them to a single monthly dataset over **2005-01 to 2024-12** (240 obs), computes the derived columns the reaction function needs, and writes a single Parquet to `country/south_africa/data/processed/sarb_l1_dataset_v1.parquet`.

The loader is the bridge between the raw immutable inputs and the estimator. Every choice that converts raw data into regression inputs lives here.

## Public API

```python
from country.south_africa.src.data.loaders import build_sarb_l1_dataset
result = build_sarb_l1_dataset(raw_dir, processed_dir, write_parquet=True)
# result.data       -> pd.DataFrame, 240 rows, indexed on date
# result.metadata   -> dict capturing every design choice
```

CLI: `python country/south_africa/src/data/loaders.py` resolves paths relative to its own location and writes the Parquet.

## Inputs expected in `data/raw/`

| File | Frequency | Columns | Source |
|---|---|---|---|
| `fed_funds_FRED_FEDFUNDS.csv` | monthly | `date, fed_funds` | FRED `FEDFUNDS` |
| `sarb_repo_rate_OECD_FRED_plus_manual_2024.csv` | monthly | `date, sarb_repo` | FRED `IRSTCB01ZAM156N` + 2024 hand-roll |
| `sa_cpi_yoy_FRED_CPALTT01ZAM659N.csv` | monthly | `date, cpi_yoy` | FRED `CPALTT01ZAM659N` |
| `zar_reer_OECD_FRED_CCRETT01ZAM661N.csv` | monthly | `date, zar_reer` | FRED `CCRETT01ZAM661N` |
| `sa_real_gdp_quarterly_FRED_NGDPRSAXDCZAQ.csv` | quarterly | `date, real_gdp_zar_mn` | FRED `NGDPRSAXDCZAQ` |
| `SA_4Q-ahead_inflation_expectations.csv` | quarterly | `date, ber_avg_1y` | BER survey, user-curated |

All monthly CSVs must extend back to **2004-01-01** to provide the 12-month history needed for `delta_zar_reer` and the 1-month history needed for `i_lag1` at the sample start.

## Output schema (`sarb_l1_dataset_v1.parquet`)

| Column | Role | Definition |
|---|---|---|
| `i_t` | target | SARB repo rate, monthly average % |
| `i_lag1` | regressor (persistence) | `i_t.shift(1)` |
| `pi_gap` | regressor (inflation gap) | `e_pi_t1y - 4.5` |
| `output_gap_nowcast` | regressor (output gap) | one-sided HP gap on log GDP, percent |
| `delta_zar_reer` | regressor (FX channel) | `100 × (log(REER_t) - log(REER_{t-12}))` |
| `fed_funds` | regressor (foreign rate) | US Fed funds rate, % |
| `gfc_dummy` | exogenous control | 1 for 2008-10 to 2009-09, else 0 |
| `cpi_yoy` | reference | realised inflation, not used in regression |
| `e_pi_t1y` | reference | BER 1Y-ahead expectation, raw |
| `zar_reer` | reference | level of REER, not used directly |

## Design choices

### 1. Inflation expectations: BER, average across 4 groups, 1Y-ahead

Following the methodology discussion: the BER Inflation Expectations Survey publishes inflation expectations for the current calendar year, the next year (year +1), the year after (year +2), and 5 years ahead. We use the **next year** expectation as the proxy for `E_t π_{t+4Q}`, and the **average across the four groups** (financial analysts, business, trade unions, households) as the headline measure.

This matches Aron-Muellbauer (2007) and the published SA monetary economics literature. Robustness check via Spec B (analysts only) is supported by simply swapping the `ber_avg_1y` column for an analyst-only column in the CSV.

### 2. Quarterly → monthly resampling for expectations

BER releases quarterly (Mar / Jun / Sep / Dec). The loader uses a **step function (forward fill)** — the value from each quarterly release applies until the next release is published. Convention: the user encodes the date column with the period the value first applies to.

The `ffill(limit=2)` is deliberate: it propagates each quarterly value to at most two subsequent months. If a quarter is missing, the next quarter's value is not back-filled to fill the gap — instead, a NaN appears and `_validate` raises. This protects against silent data extension when the BER series is incomplete.

### 3. Real-time output gap on quarterly GDP

The output gap is computed by running a **one-sided HP filter** on quarterly real GDP (`λ = 1600`, standard for quarterly), then linearly interpolating the resulting quarterly gap to monthly.

For each quarter t, the HP filter is re-estimated using only data through quarter t. This avoids the look-ahead bias of the standard two-sided HP filter and reproduces what SARB could plausibly have observed at time t.

A minimum of 16 quarters (4 years) of GDP history is required before the gap is computed. With the FRED series starting in 1993-Q1, this means gaps are available from 1997-Q1 onwards — well before the sample start of 2005-Q1.

**Caveats.** One-sided HP is endpoint-biased compared to full-sample. The Hamilton (2018) filter is a defensible alternative. For pilot-stage estimation this is acceptable; the structural ranking of dissonance episodes does not depend critically on the gap definition. Document in the data appendix as a known limitation.

### 4. Monthly interpolation of the quarterly gap

The quarterly gap is linearly interpolated to monthly. Each month within a quarter receives a weighted average of the surrounding quarter-end gaps. Equivalent to assuming the underlying gap evolves smoothly within the quarter.

Alternative: hold the quarterly gap constant within the quarter (step function). The chosen linear approach is smoother and avoids artificial discontinuities at quarter boundaries.

### 5. Δ ZAR REER = 12-month log difference

`delta_zar_reer = 100 × (log(REER_t) - log(REER_{t-12}))`

Captures the *year-on-year real appreciation*. Convention: positive = real appreciation (currency stronger).

Aron-Muellbauer used `Δ log REER` over various horizons; the 12-month differencing handles seasonality cleanly and is robust to monthly noise.

Robustness alternative: 3-month change, or first difference of log REER. Easy to swap in the loader.

### 6. GFC dummy

`gfc_dummy = 1` for 2008-10-01 through 2009-09-01. These are the **Stats SA recession dates** for South Africa, not US dates. SA-specific is appropriate since this is the SARB reaction function.

Justification: during deep recessions the standard reaction function breaks down (banks de-lever, transmission collapses, panic responses dominate). Dummying these months out avoids contaminating the structural parameters with crisis dynamics.

Trade-off: dummies absorb information. We're not using them for OOS prediction. Robustness check: re-estimate without the dummy and report whether α̂, β̂ change meaningfully.

### 7. Sample window: 2005-01 to 2024-12

Aligns with the pre-registered framework. 2005 chosen because:
- SARB inflation targeting framework introduced in 2000, fully embedded by 2005
- BER survey published continuously from 2000 but methodology stabilised by 2005
- 240 monthly observations is a defensible sample size for OLS+HAC estimation

### 8. Pre-sample data extension to 2004-01

Required for:
- `i_lag1` at 2005-01-01 needs `i_t` at 2004-12-01 (1 month pre-sample)
- `delta_zar_reer` at 2005-01-01 needs `REER` at 2004-01-01 (12 months pre-sample)

The four monthly CSVs include 2004 data sourced from the same FRED series. The loader silently uses this pre-sample data and then trims to the canonical window.

### 9. SARB target = 4.5%

Midpoint of the 3-6% inflation target band, official since 2000. Re-anchoring to 4.5% at every observation is the simplest defensible specification (constant target, time-invariant). 

Aron-Muellbauer used 4.5 with some sensitivity analysis around 5.0; we follow that convention.

A more sophisticated treatment would use the **time-varying perceived target** (the implicit centre of SARB's communicated path) — defer for now, flag as a refinement.

## Validation

Before writing the Parquet, the loader runs `_validate(df)` which checks:
- All required regressor columns exist
- Exactly 240 rows (matches sample window)
- No NaN values in any required column

If any check fails the loader raises `ValueError` with the first offending row, never silently writes partial data.

## Failure modes & handling

| Scenario | Behaviour |
|---|---|
| Missing CSV file | `FileNotFoundError` with the expected path and (for BER) the expected schema |
| Wrong column name in a CSV | `KeyError` listing the columns actually found |
| BER series doesn't cover full 2005-2024 | `_validate` raises (NaN in `pi_gap`) |
| Pre-sample 2004 data missing | `_validate` raises (NaN in `i_lag1` or `delta_zar_reer`) at 2005-01 |
| Comments at top of CSV (lines starting `#`) | Silently skipped via `comment='#'` |

## Reproducibility

The loader is deterministic. The pipeline from `raw/` → Parquet has no random state, no external API calls, no caching.

Two ways the output can change despite identical raw CSVs:
1. **statsmodels version drift** affecting `hpfilter` — pinned in environment
2. **pandas resampling behaviour** changing — pinned in environment

Both are version-pinned in the project requirements.

## Extensions (post-pilot)

- **Alternative output gap measures**: Hamilton filter, SARB QPM gap, IMF estimates → add as separate columns, switch via metadata flag
- **Robust expectations measures**: analyst-only, Reuters poll, blended series → swap CSV
- **Vintage-aware data**: replace each CSV with a real-time vintage panel (ALFRED), revise loader to take a vintage date parameter → much larger lift
- **Multi-country generalisation**: deliberately not done — this loader is SARB-bespoke
