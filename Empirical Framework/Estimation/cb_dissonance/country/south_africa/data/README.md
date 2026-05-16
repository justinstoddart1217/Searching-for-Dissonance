# SARB Reaction Function — Raw Data

Pilot dataset for Level 1 (reaction function) estimation. All series **2005-01 to 2024-12** (240 monthly obs) unless noted. Sourced via FRED's public web interface.

## Files

| File | Variable | Source | Notes |
|---|---|---|---|
| `fed_funds_FRED_FEDFUNDS.csv` | `fed_funds` (i_foreign) | BoG via FRED `FEDFUNDS` | Monthly average. Clean. |
| `sarb_repo_rate_OECD_FRED_plus_manual_2024.csv` | `sarb_repo` (i_t) | OECD MEI via FRED `IRSTCB01ZAM156N` (2005–2023) + manual 2024 from MPC announcements | OECD ended series Dec 2023. 2024 values weighted by days at each rate (Jan–Aug 8.25, Sep cut to 8.00 on 19 Sep, Nov cut to 7.75 on 21 Nov). **Verify against Bloomberg `SARPRATE Index` before serious use.** |
| `sa_cpi_yoy_FRED_CPALTT01ZAM659N.csv` | `cpi_yoy` (π_t) | OECD MEI via FRED `CPALTT01ZAM659N` | Headline CPI, YoY % (transformation GY). Realised inflation only — **not** the 4Q-ahead expectation. |
| `zar_reer_OECD_FRED_CCRETT01ZAM661N.csv` | `zar_reer` (q_t) | OECD MEI via FRED `CCRETT01ZAM661N` | CPI-based REER, Index 2015=100. Increase = real appreciation. Loader will compute YoY delta for Δq. |
| `sa_real_gdp_quarterly_FRED_NGDPRSAXDCZAQ.csv` | `real_gdp_zar_mn` (→ ỹ_t) | IMF IFS via FRED `NGDPRSAXDCZAQ` | Quarterly, seasonally adjusted, 1993-Q1 to 2025-Q4. Loader will interpolate to monthly and one-sided HP filter to extract `output_gap_nowcast`. |

## Still to source — Justin investigating

**SA 4Q-ahead inflation expectations (`E_t π_{t+4}`)** — the hardest series, source-of-truth for `pi_gap = E_t π_{t+4} − 4.5`. Ranked options:

1. **BER Inflation Expectations Survey** (Stellenbosch) — quarterly since 2000, real-time by construction. Aron-Muellbauer's preferred source. Best if Ninety One has BER data feed.
2. **SARB MPC's published forecasts** — in MPC statements since 2007, tabular form, real-time. Slight gap pre-2007.
3. **Reuters/Bloomberg poll consensus** — vendor history, may have early-year gaps.
4. **Naïve AR(1) on realised inflation** — fallback for missing months. Documents the substitution in the data appendix.

For the pilot, even BER quarterly interpolated to monthly with AR(1) gap-fill is defensible. Aron-Muellbauer themselves used a survey-based measure.

## Caveats

- All FRED OECD series come via the OECD Main Economic Indicators database with citation requirement. License terms permit research use.
- SARB repo rate for 2024 is manually compiled from MPC announcements — **verify before quoting**.
- The OECD CPI series is the *all-items* CPI, not the official Stats SA headline CPI. Levels and YoY should match closely but minor methodological differences exist. For thesis-quality work, swap to Stats SA `CPI Headline (P0141)` once Bloomberg pull is available.
- The OECD CB rate series uses OECD's reweighting (note the fractional intra-month values when rate changes mid-month). For comparison to SARB-published rates (typically end-of-month integer), expect small differences in months with MPC decisions.
- Real GDP is monthly-coverage requires interpolation. The output-gap calculation in the loader will be a one-sided HP filter (lambda=14400 quarterly) — coherent with real-time information but with the usual end-point bias caveat.

## Next step

Build `country/south_africa/data/loaders.py` to:
1. Read these CSVs.
2. Harmonise to monthly index 2005-01 to 2024-12.
3. Compute derived columns: `i_lag1`, `pi_gap`, `delta_zar_reer` (12-month log diff), `output_gap_nowcast` (one-sided HP on interpolated monthly GDP), `gfc_dummy` (2008-10 to 2009-09 = 1).
4. Write single Parquet `sarb_l1_dataset_v1.parquet` keyed on date.

Then run `SARBReactionFunction().fit(data)` and compare θ̂ against Aron-Muellbauer (2007) coefficients.
