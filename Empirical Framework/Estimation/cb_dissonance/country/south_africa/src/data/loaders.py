from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm


SAMPLE_START = '2005-01-01'
SAMPLE_END = '2024-12-01'
GFC_START = '2008-10-01'
GFC_END = '2009-09-01'
SARB_TARGET = 4.5
HP_LAMBDA_QUARTERLY = 1600
HP_MIN_HISTORY_QUARTERS = 16
EXPECTED_N_OBS = 240
OUTPUT_PARQUET = 'sarb_l1_dataset_v1.parquet'

FILES = {
    'fed_funds': ('fed_funds_FRED_FEDFUNDS.csv', 'fed_funds'),
    'sarb_repo': ('sarb_repo_rate_OECD_FRED_plus_manual_2024.csv', 'sarb_repo'),
    'cpi_yoy': ('sa_cpi_yoy_FRED_CPALTT01ZAM659N.csv', 'cpi_yoy'),
    'zar_reer': ('zar_reer_OECD_FRED_CCRETT01ZAM661N.csv', 'zar_reer'),
    'gdp_q': ('sa_real_gdp_quarterly_FRED_NGDPRSAXDCZAQ.csv', 'real_gdp_zar_mn'),
    'ber_e_pi': ('SA_4Q-ahead_inflation_expectations.csv', '4Q Inflation Expectations'),
}


@dataclass(frozen=True)
class SARBLevel1Dataset:
    data: pd.DataFrame
    metadata: dict


def _read_series(path: Path, value_col: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw input: {path}")
    df = pd.read_csv(path, comment='#')
    if 'date' not in df.columns:
        raise KeyError(f"{path.name}: missing 'date' column. Got: {list(df.columns)}")
    if value_col not in df.columns:
        raise KeyError(f"{path.name}: missing '{value_col}' column. Got: {list(df.columns)}")
    # Raw CSVs are mixed: some use DD/MM/YYYY (slash-separated, e.g. FRED
    # downloads, BER survey), others ISO YYYY-MM-DD (dash-separated). Parse
    # slash dates with dayfirst=True; let ISO dates parse with the default.
    date_str = df['date'].astype(str)
    if date_str.str.contains('/').any():
        df['date'] = pd.to_datetime(date_str, dayfirst=True)
    else:
        df['date'] = pd.to_datetime(date_str)
    return df.set_index('date').sort_index()[value_col]


def _load_monthly(raw_dir: Path, key: str) -> pd.Series:
    fname, col = FILES[key]
    s = _read_series(raw_dir / fname, col)
    s.index = pd.to_datetime(s.index).to_period('M').to_timestamp()
    s.name = key
    return s


def _load_quarterly_gdp(raw_dir: Path) -> pd.Series:
    fname, col = FILES['gdp_q']
    s = _read_series(raw_dir / fname, col)
    s.name = 'real_gdp_zar_mn'
    return s


def _load_inflation_expectations(raw_dir: Path) -> pd.Series:
    fname, col = FILES['ber_e_pi']
    path = raw_dir / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw input: {path}\n"
            f"Expected CSV schema:\n"
            f"  date,{col}\n"
            f"  2005-01-01,4.6\n"
            f"  2005-04-01,4.7\n"
            f"  ...\n"
            f"Frequency: quarterly. Date = first month of period the value applies to.\n"
            f"Value: BER survey, average across 4 groups, 1-year-ahead expected CPI YoY (%)."
        )
    s = _read_series(path, col)
    monthly_index = pd.date_range(s.index.min(), s.index.max() + pd.offsets.MonthBegin(2), freq='MS')
    s = s.reindex(monthly_index).ffill(limit=2)
    s.name = 'e_pi_t1y'
    return s


def _real_time_hp_gap_quarterly(gdp_q: pd.Series, lambda_hp: float, min_history: int) -> pd.Series:
    log_gdp = np.log(gdp_q)
    gap_q = pd.Series(np.nan, index=gdp_q.index, name='gap_q')
    for t in range(min_history, len(log_gdp)):
        sample = log_gdp.iloc[: t + 1]
        cycle, _ = sm.tsa.filters.hpfilter(sample, lamb=lambda_hp)
        gap_q.iloc[t] = float(cycle.iloc[-1]) * 100.0
    monthly_index = pd.date_range(gap_q.index.min(), gap_q.index.max() + pd.offsets.MonthBegin(2), freq='MS')
    gap_m = gap_q.reindex(monthly_index).interpolate(method='linear', limit_direction='forward')
    gap_m.name = 'output_gap_nowcast'
    return gap_m


def _delta_reer_yoy(reer: pd.Series) -> pd.Series:
    log_reer = np.log(reer)
    delta = 100.0 * (log_reer - log_reer.shift(12))
    delta.name = 'delta_zar_reer'
    return delta


def _build_gfc_dummy(index: pd.DatetimeIndex) -> pd.Series:
    dummy = pd.Series(0.0, index=index, name='gfc_dummy')
    mask = (index >= pd.Timestamp(GFC_START)) & (index <= pd.Timestamp(GFC_END))
    dummy.loc[mask] = 1.0
    return dummy


def _validate(df: pd.DataFrame) -> None:
    required = ['i_t', 'i_lag1', 'pi_gap', 'output_gap_nowcast', 'delta_zar_reer', 'fed_funds', 'gfc_dummy']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(df) != EXPECTED_N_OBS:
        raise ValueError(f"Expected {EXPECTED_N_OBS} rows, got {len(df)}")
    for col in required:
        n_na = df[col].isna().sum()
        if n_na > 0:
            first_na = df.index[df[col].isna()][0]
            raise ValueError(f"Column '{col}' has {n_na} NaN; first at {first_na.date()}")


def build_sarb_l1_dataset(
    raw_dir: Path | str,
    processed_dir: Path | str,
    write_parquet: bool = True,
) -> SARBLevel1Dataset:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    fed_funds = _load_monthly(raw_dir, 'fed_funds')
    sarb_repo = _load_monthly(raw_dir, 'sarb_repo')
    cpi_yoy = _load_monthly(raw_dir, 'cpi_yoy')
    zar_reer = _load_monthly(raw_dir, 'zar_reer')
    gdp_q = _load_quarterly_gdp(raw_dir)
    e_pi_t1y = _load_inflation_expectations(raw_dir)

    output_gap = _real_time_hp_gap_quarterly(gdp_q, HP_LAMBDA_QUARTERLY, HP_MIN_HISTORY_QUARTERS)
    delta_reer = _delta_reer_yoy(zar_reer)

    full_index = pd.date_range('2004-01-01', '2024-12-01', freq='MS')
    df = pd.DataFrame(index=full_index)
    df.index.name = 'date'
    df['i_t'] = sarb_repo
    df['cpi_yoy'] = cpi_yoy
    df['e_pi_t1y'] = e_pi_t1y
    df['zar_reer'] = zar_reer
    df['delta_zar_reer'] = delta_reer
    df['fed_funds'] = fed_funds
    df['output_gap_nowcast'] = output_gap
    df['i_lag1'] = df['i_t'].shift(1)
    df['pi_gap'] = df['e_pi_t1y'] - SARB_TARGET
    df['gfc_dummy'] = _build_gfc_dummy(df.index)

    df = df.loc[SAMPLE_START:SAMPLE_END].copy()

    _validate(df)

    metadata = {
        'sample_start': SAMPLE_START,
        'sample_end': SAMPLE_END,
        'n_obs': int(len(df)),
        'sarb_target_pct': SARB_TARGET,
        'gfc_dummy_window': (GFC_START, GFC_END),
        'hp_lambda_quarterly': HP_LAMBDA_QUARTERLY,
        'hp_min_history_quarters': HP_MIN_HISTORY_QUARTERS,
        'hp_strategy': 'one-sided real-time HP on quarterly GDP, then linear interp of gap to monthly',
        'expectations_horizon': '1Y (BER survey, average across financial analysts, business, trade unions, households)',
        'expectations_resampling': 'quarterly -> monthly via step function (ffill within quarter, limit=2)',
        'reer_delta_def': '100 * (log(REER_t) - log(REER_{t-12}))',
        'regressors': ['i_lag1', 'pi_gap', 'output_gap_nowcast', 'delta_zar_reer', 'fed_funds'],
        'target': 'i_t',
        'exogenous_controls': ['gfc_dummy'],
    }

    if write_parquet:
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = processed_dir / OUTPUT_PARQUET
        df.to_parquet(out_path)

    return SARBLevel1Dataset(data=df, metadata=metadata)


if __name__ == '__main__':
    here = Path(__file__).resolve()
    country_root = here.parent.parent.parent
    raw_dir = country_root / 'data' / 'raw'
    processed_dir = country_root / 'data' / 'processed'
    result = build_sarb_l1_dataset(raw_dir, processed_dir)
    print(f"Wrote {result.metadata['n_obs']} rows to {processed_dir / OUTPUT_PARQUET}")
    print()
    print("Summary statistics:")
    print(result.data.describe().T.round(3))
    print()
    print("First 5 rows:")
    print(result.data.head().round(3))
    print()
    print("Last 5 rows:")
    print(result.data.tail().round(3))
