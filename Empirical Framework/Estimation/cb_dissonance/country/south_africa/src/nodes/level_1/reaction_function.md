# SARB Reaction Function — Implementation Documentation

This document explains the code in `reaction_function.py` — what it
estimates, why each design choice was made, and how the output is
consumed downstream.

---

## 1. Purpose

The module estimates the **South African Reserve Bank's interest rate
reaction function** — the rule that maps the macroeconomic state to
SARB's repo rate decision. The implementation follows the
Aron-Muellbauer (2007) specification, which is the canonical
open-economy CGG forward-smoothed Taylor rule for SARB.

The fitted reaction function produces three outputs consumed by the
downstream dissonance framework:

1. **`i*_t`** — the *rule-implied* repo rate at every date in the
   calibration and out-of-sample windows. This is what the rule
   says SARB *should have done* given the data.
2. **`ε_t = i_t − i*_t`** — the Taylor residual. The deviation of
   actual policy from the rule's prescription. This series is the
   consistency function for the L1↔L3a edge.
3. **`θ̂`** — the structural coefficient vector
   `{α̂, β̂, ρ̂, γ̂, δ̂}`. Describes SARB's "type" — hawk vs dove,
   gradualist vs activist, FX-sensitive vs domestically-focused.

The L1↔L3a correlation test, consistency function, and dissonance
function all consume the `SARBReactionFunctionFit` output of this
module.

---

## 2. Theoretical Background

### 2.1 The Taylor Rule Family

The Taylor (1993) rule is the canonical empirical description of how
modern central banks set policy rates:

```
i_t = r* + π* + α(π_t − π*) + β·ỹ_t
```

The bank raises rates above neutral when inflation overshoots target
(α > 0) and when the economy runs hot (β > 0). The **Taylor
principle** — α > 1 — is the stability condition: the bank must move
*nominal* rates more than 1-for-1 with inflation to actually move
*real* rates. If α < 1, the rule produces equilibrium indeterminacy.

### 2.2 CGG (1999, 2000) — Forward-Looking, Smoothed

Two refinements of the original Taylor specification are now standard.

**Forward-looking inflation.** Monetary policy operates with lags of
4-8 quarters. By the time realised inflation arrives, policy can
only respond to what's coming next. Banks therefore react to expected
inflation, not contemporaneous inflation:

```
i_t = r* + α·E_t π_{t+4} + β·E_t ỹ_t
```

**Smoothing.** Banks adjust rates gradually, not abruptly. The
smoothed version weights the lagged rate:

```
i_t = ρ·i_{t-1} + (1 − ρ)·[r* + α·E_t π_{t+4} + β·E_t ỹ_t]
```

The bracketed term is the **long-run target rate** — where the bank
wants to be given current conditions. The `ρ·i_{t-1}` term anchors
at **where the bank currently is**. Each meeting, the bank takes a
weighted average of those two. ρ is typically estimated at 0.7–0.9
across DM and EM banks.

### 2.3 Aron-Muellbauer (2007) — SARB-Specific Extension

For SARB specifically, two open-economy regressors matter materially:

- **`Δq_t`** (YoY change in REER) — SARB systematically responds to
  rand weakness because pass-through to inflation is meaningful for
  South Africa. This is the Calvo-Reinhart (2002) "fear of floating"
  channel.
- **`fed_funds_t`** — SARB cannot stray too far from US monetary
  conditions without producing capital flow distortions. This
  captures the Rey (2013) dilemma — even floating-FX EM banks face
  binding global-financial-cycle constraints.

The full SARB specification, which is what the code estimates:

```
i_t = ρ·i_{t-1} + (1 − ρ)·[r* + α·(E_t π_{t+4} − π*)
      + β·E_t ỹ_t + γ·Δq_t + δ·fed_funds_t] + ε_t
```

---

## 3. SARB-Specific Specification Choices

Every constant in the class corresponds to a concrete empirical or
institutional fact about SARB. They are exposed as class-level
attributes (uppercase, no leading underscore) so they can be inspected,
audited, and referenced in the thesis chapter.

### 3.1 `PI_TARGET = 4.5`

SARB's inflation target band has been **3–6%** since 2000. The
operational target — what SARB actually aims at — is the midpoint,
**4.5%**. The MPC's statements consistently reference the midpoint as
the anchor.

This constant lets the inflation gap be computed as
`pi_gap = E_t π_{t+4} − 4.5` upstream (in the data pipeline). If SARB
eventually moves to a point target (proposed but not implemented as
of 2024), this constant is the single change point.

### 3.2 `GFC_START = '2008-10'`, `GFC_END = '2009-09'`

The South African recession during the GFC ran four consecutive
quarters: **2008-Q4 through 2009-Q3**, per the Stats SA Q3 2009 GDP
release. The dummy is set to 1 during these months, zero otherwise.

These dates differ from the US recession (2007-Q4 to 2009-Q2) and the
eurozone recession (2008-Q2 to 2009-Q2). Using US dates for SARB
would mis-time the dummy — particularly bad because the SARB rate
response to the GFC was lagged relative to the Fed (first SARB cut in
December 2008, vs. Fed cuts starting September 2007).

### 3.3 `DEFAULT_CALIBRATION_WINDOW = ('2005-01', '2014-12')`

Ten years, ending before the 2015–16 commodity cycle stress and the
2018 emerging-market sell-off. Starting in 2005 avoids the early-IT
(1999–2004) regime when SARB was still building credibility under the
new inflation-targeting framework. By 2005, the regime was
empirically stable.

### 3.4 `DEFAULT_OOS_WINDOW = ('2015-01', '2024-12')`

The full out-of-sample window for the dissonance series. Includes the
2015–16 commodity crash, the 2018 EM sell-off, COVID 2020, and the
2022–23 hiking cycle — all of which should produce dissonance flags
in the framework.

### 3.5 Regressor Names

The class constant `REGRESSORS` uses SARB-specific names rather than
generic placeholders:

| Generic name | SARB name            | What it is                               |
|--------------|----------------------|------------------------------------------|
| `i_lag1`     | `i_lag1`             | Lagged SARB repo rate                    |
| `pi_gap`     | `pi_gap`             | `E_t π_{t+4} − PI_TARGET`                |
| `output_gap` | `output_gap_nowcast` | Output gap (real-time vintage)           |
| `delta_q`    | **`delta_zar_reer`** | YoY change in ZAR REER, broad basket     |
| `i_foreign`  | **`fed_funds`**      | US federal funds rate                    |

This makes the data contract explicit. The data loader (built
separately) constructs a DataFrame with these exact column names; the
estimator reads them by name. If the contract is ever violated (e.g.,
a different REER index is substituted), the explicit naming makes the
substitution visible at the column level.

### 3.6 `HAC_MAXLAGS = 6`

The Newey-West lag length for the HAC covariance estimator. Six
months is the conventional choice for monthly macro data, large
enough to capture meaningful serial correlation in the residuals
(driven by overlapping monetary transmission lags) without
over-fitting the covariance.

### 3.7 `RHO_GUARD_THRESHOLD = 0.95`

When ρ̂ approaches 1, the structural coefficient recovery becomes
unstable because the divisor `(1 − ρ̂)` approaches zero. We flag any
window where ρ̂ ≥ 0.95 and set the structural coefficients to NaN.
The linearised θ̂ vector is still reported intact — those parameters
are well-estimated, the problem is the nonlinear recovery.

---

## 4. Code Architecture

The module exposes two classes:

1. **`SARBReactionFunctionFit`** — a frozen dataclass holding the
   results of a single estimation.
2. **`SARBReactionFunction`** — the estimator itself. Its `.fit(...)`
   method takes a DataFrame and returns a `SARBReactionFunctionFit`.

The split mirrors a common pattern in econometric software:
estimator + fitted result. The estimator carries the specification
(constants); the fit carries the outputs. Multiple fits can be
produced from the same estimator instance (with different windows or
settings) without state contamination.

### 4.1 Why `frozen=True`?

A fitted result is a record. Once an estimation completes, nothing
downstream should mutate the result. The frozen decorator prevents
accidental mutation: an attempt to write `fit.residuals = something`
raises `FrozenInstanceError`. This is cheap insurance against silent
corruption further down the dissonance pipeline.

### 4.2 Why `eq=False`?

The default dataclass `__eq__` compares fields element-wise. For the
pandas Series fields (`i_star`, `residuals`), element-wise comparison
returns a Series of booleans, not a single bool — which breaks
`fit_a == fit_b` and produces a `ValueError` ("The truth value of a
Series is ambiguous"). Disabling auto-generated equality avoids the
issue; we don't need to compare fit objects in any downstream code.

---

## 5. The `.fit()` Method — Workflow

Signature:

```python
fit(
    data: pd.DataFrame,
    calibration_window: tuple[str, str] | None = None,
    oos_window: tuple[str, str] | None = None,
    gfc_dummy: bool = True,
    method: str = 'ols',
) -> SARBReactionFunctionFit
```

Executes the full estimation pipeline in eight steps.

### Step 1: Method validation

Only `method='ols'` is currently supported. GMM is deferred — its
interface will plug into the same `.fit()` signature when implemented
in a later iteration.

### Step 2: Window defaults

If `calibration_window` or `oos_window` are `None`, the SARB defaults
are used. Routine use is just:

```python
rf = SARBReactionFunction()
fit = rf.fit(data)
```

The full signature is exposed for sensitivity analysis — shift the
calibration window earlier or later to test robustness, particularly
across the GFC regime break.

### Step 3: Calibration data preparation

`_prepare_data(...)` slices the calibration window, builds the
regression matrix, optionally adds the GFC dummy, adds the OLS
constant, drops rows with any NaN, and enforces a minimum sample
size of 30 observations. Returns `(y_calib, X_calib)`.

### Step 4: OLS with HAC standard errors

```python
sm.OLS(y_calib, X_calib).fit(
    cov_type='HAC',
    cov_kwds={'maxlags': 6},
)
```

OLS gives the point estimates. HAC (Newey-West) gives standard errors
robust to serial correlation in the residuals — which is the norm in
Taylor rule residuals even with smoothing, because monetary
transmission produces overlapping shocks.

### Step 5: Extract linearised coefficients

`_extract_linear_params(...)` pulls `theta_linear` (the raw OLS
coefficients) and `theta_linear_se` (their HAC standard errors) from
the fitted result.

### Step 6: Recover structural coefficients

`_recover_structural(...)` converts the linearised parameters to
structural form via `α̂ = θ̂_1/(1−ρ̂)` etc., and computes delta-method
standard errors. If ρ̂ ≥ 0.95, structural estimates are flagged
unreliable and set to NaN.

### Step 7: OOS prediction

The calibration θ̂ is applied to OOS state data via
`ols_result.predict(X_oos)`. This is the central methodological
move: parameters fitted on 2005–2014 predict the policy rate on
2015–2024. Deviations of actual `i_t` from `i*_t` on the OOS window
are the dissonance signal.

### Step 8: Assemble output

`i_star` and `residuals` are concatenated over both windows and
sorted by date. Diagnostics include in-sample R², adjusted R²,
Durbin-Watson statistic, and the OOS missing-data fraction. Metadata
records the method, specification name, and SARB-specific constants
used.

---

## 6. Linearisation Detail

The structural form

```
i_t = ρ·i_{t-1} + (1−ρ)·[r* + α·gap + β·ỹ + γ·Δq + δ·fed] + ε_t
```

is **nonlinear in parameters** because of the `(1−ρ)·α` etc.
products. Standard OLS requires linearity in parameters. The trick is
to estimate the linearised form

```
i_t = c + ρ·i_{t-1} + θ_1·gap + θ_2·ỹ + θ_3·Δq + θ_4·fed + ε_t
```

where `c = (1−ρ)·r*`, `θ_1 = (1−ρ)·α`, `θ_2 = (1−ρ)·β`, etc. This is
linear in `(c, ρ, θ_1, ..., θ_4)`. After OLS gives θ̂, we recover the
structural coefficients algebraically:

```
α̂ = θ̂_1 / (1 − ρ̂)
β̂ = θ̂_2 / (1 − ρ̂)
γ̂ = θ̂_3 / (1 − ρ̂)
δ̂ = θ̂_4 / (1 − ρ̂)
```

(The constant `c` absorbs `(1−ρ)·r*`. The neutral real rate `r*` is
not separately identified unless an external value is imposed.)

The mapping is encoded in two class constants:
- **`LINEAR_TO_COL`** maps `theta_j → regressor column name`. Used by
  `_extract_linear_params` to pull the right column from the OLS
  result.
- **`LINEAR_TO_STRUCT`** maps `theta_j → structural coefficient name`.
  Used by `_recover_structural` to label the recovered values.

Keeping these as class constants means the mapping is auditable in
one place rather than scattered across the code.

---

## 7. Delta-Method Standard Errors

Once we have `α̂ = f(θ̂_1, ρ̂) = θ̂_1 / (1 − ρ̂)`, we need a standard
error on `α̂`. OLS gives standard errors on θ̂_1 and ρ̂ directly, but
α̂ is a nonlinear transformation. The **delta method** gives the SE
on the transformation:

```
Var(α̂) ≈ (∂f/∂θ_1)² · Var(θ_1)
       + (∂f/∂ρ)²  · Var(ρ)
       + 2 · (∂f/∂θ_1) · (∂f/∂ρ) · Cov(θ_1, ρ)
```

The partial derivatives:

```
∂f/∂θ_1 = 1 / (1 − ρ)
∂f/∂ρ   = θ_1 / (1 − ρ)²
```

The variance and covariance terms come from the OLS covariance matrix
(`ols_result.cov_params()`). Crucially, the **covariance term is not
zero** — θ̂_1 and ρ̂ come from the same regression and are
correlated, so dropping the covariance term would understate the SE
on α̂.

```
SE(α̂) = sqrt(Var(α̂))
```

This is computed for each structural coefficient (α, β, γ, δ) inside
`_recover_structural`. The SE on ρ̂ itself comes directly from OLS —
no transformation needed.

### 7.1 Why the Delta Method, Not Bootstrap?

For 120 monthly observations and four structural coefficients, the
delta method gives closed-form, reproducible SEs in milliseconds.
Bootstrap would be more flexible (handles weird small-sample
distributions) but adds a stochastic element to a deterministic
pipeline and costs orders of magnitude more compute. The delta method
is the standard choice in the Taylor rule literature.

### 7.2 Numerical Safeguard

The delta-method variance can occasionally come back negative if the
HAC covariance matrix has small negative eigenvalues from numerical
noise (more common in small samples). The code checks and reports NaN
in that case rather than producing a complex-valued SE.

---

## 8. The ρ̂ Guard

When the policy rate is highly persistent in the calibration window,
ρ̂ comes back close to 1. The divisor `(1 − ρ̂)` in the structural
recovery approaches zero, and structural SEs blow up.

The guard threshold is `RHO_GUARD_THRESHOLD = 0.95`. Above this:

- `theta_hat['alpha']`, `['beta']`, `['gamma']`, `['delta']` are set
  to `NaN`.
- Corresponding SE entries are set to `NaN`.
- `structural_reliable = False` propagates to the diagnostics.
- The linearised θ̂ vector is **preserved intact** — these
  coefficients are well-estimated, only the nonlinear recovery
  fails.

This matters operationally: a window where the SARB rate behaves like
a near-random walk (e.g., extended hold periods, or a sample
dominated by the 2020 effective lower bound for some banks) will
flag, and downstream code can still use the linearised θ̂ for
correlation testing while marking the structural interpretation as
unreliable.

---

## 9. The Output: `SARBReactionFunctionFit`

A frozen dataclass with eight fields:

### 9.1 `theta_hat`

Dict of structural coefficients: `{'rho', 'alpha', 'beta', 'gamma', 'delta'}`.
NaN-filled for α/β/γ/δ when `structural_reliable = False`. ρ̂ is
always populated.

### 9.2 `theta_se`

Dict of delta-method standard errors with `_se` suffix:
`{'rho_se', 'alpha_se', 'beta_se', 'gamma_se', 'delta_se'}`.

### 9.3 `theta_linear`

Dict of raw OLS coefficients:
`{'c', 'rho', 'theta_1', 'theta_2', 'theta_3', 'theta_4'}`.
Always populated, even when structural recovery fails the guard.

### 9.4 `theta_linear_se`

Dict of OLS standard errors on the linearised coefficients (same
keys with `_se` suffix).

### 9.5 `i_star`

A pandas Series of the rule-implied SARB repo rate, indexed by date,
spanning the calibration window (fitted values) followed by the OOS
window (out-of-sample predictions). Sorted by date.

### 9.6 `residuals`

A pandas Series of `ε_t = i_t − i*_t`, same index as `i_star`. This
is the consistency function `R^{L1↔L3a}_t` that feeds the L1↔L3a
dissonance metric.

### 9.7 `diagnostics`

Dict with keys:

| Key                     | Type   | Meaning                                       |
|-------------------------|--------|-----------------------------------------------|
| `r_squared`             | float  | In-sample R² on the calibration window        |
| `adj_r_squared`         | float  | Adjusted R² (penalises regressor count)       |
| `durbin_watson`         | float  | Residual autocorrelation diagnostic           |
| `n_obs_calibration`     | int    | Sample size after NaN-dropping                |
| `n_obs_oos`             | int    | OOS observations where prediction was made    |
| `missing_oos_fraction`  | float  | Fraction of OOS dates dropped                 |
| `structural_reliable`   | bool   | True if ρ̂ < `RHO_GUARD_THRESHOLD`             |

### 9.8 `metadata`

Dict with keys:

| Key                  | Value                                              |
|----------------------|----------------------------------------------------|
| `method`             | `'ols'`                                            |
| `calibration_window` | The tuple actually used                            |
| `oos_window`         | The tuple actually used                            |
| `gfc_dummy_used`     | True if the dummy column was applied               |
| `pi_target`          | 4.5 (recorded for audit)                           |
| `specification`      | `'Aron-Muellbauer (2007) CGG forward-smoothed'`    |

---

## 10. Helper Methods

### 10.1 `_prepare_data(data, window, *, gfc_dummy, min_obs)`

The data-wrangling layer. Takes a window argument so it can serve
both calibration and OOS preparation. Inside, it:

1. Validates required columns (`['i_t']` + `REGRESSORS`); raises
   `ValueError` with a clear message if any are missing.
2. Slices the date window inclusively. Raises `ValueError` if the
   slice is empty (catches typos in window dates).
3. Builds the regressor matrix from `REGRESSORS`.
4. Adds the GFC dummy if `gfc_dummy=True` *and* the column exists.
5. Adds the OLS constant (column of ones).
6. Drops rows with NaN in y or X (after combining).
7. Enforces `min_obs` (default 30 for calibration; OOS uses 0).

The keyword-only arguments after `*` (gfc_dummy, min_obs) prevent
positional confusion at the call site.

### 10.2 `_extract_linear_params(ols_result)`

Pulls the linearised coefficients from the OLS result by column name.
Uses `LINEAR_TO_COL` to know which OLS column corresponds to which
linearised parameter name. Returns two parallel dicts (values and
SEs). All values cast to `float` so the output is JSON-serialisable
for later persistence.

### 10.3 `_recover_structural(theta_linear, ols_result)`

The structural-recovery and delta-method core. Returns a tuple of
`(theta_hat, theta_se, structural_reliable)`. The boolean propagates
the ρ̂ guard verdict downstream into the diagnostics dict.

---

## 11. Usage Example

```python
import pandas as pd
from reaction_function import SARBReactionFunction

# Load SARB data (built upstream by the data loader)
data = pd.read_parquet(
    'country/south_africa/data/harmonised/l1_state.parquet'
)

# Instantiate and fit with SARB defaults
rf = SARBReactionFunction()
fit = rf.fit(data)

# Inspect structural coefficients
print(f"alpha = {fit.theta_hat['alpha']:.3f} "
      f"(se {fit.theta_se['alpha_se']:.3f})")
# Expected: alpha around 1.5 (Taylor principle satisfied)

print(f"rho   = {fit.theta_hat['rho']:.3f} "
      f"(se {fit.theta_se['rho_se']:.3f})")
# Expected: rho around 0.8 (typical EM smoothing)

# Inspect fit quality
print(f"R² = {fit.diagnostics['r_squared']:.3f}")
# Expected: R² > 0.85 in a clean calibration window

# Use i_star and residuals downstream for the dissonance pipeline
i_star = fit.i_star          # rule-implied repo rate, calibration + OOS
residuals = fit.residuals    # Taylor residual ε_t — input to dissonance
```

To stress-test the calibration window:

```python
# Pre-GFC only
fit_pre = rf.fit(data, calibration_window=('2005-01', '2008-09'))

# Post-GFC only
fit_post = rf.fit(data, calibration_window=('2010-01', '2014-12'))

# Compare structural coefficients across the two — if they differ
# meaningfully, the rule may have shifted across the GFC, which is
# itself a useful finding for the SA pilot chapter.
```

---

## 12. References

- Taylor, J. B. (1993). "Discretion versus policy rules in
  practice." *Carnegie-Rochester Conference Series on Public
  Policy*.
- Clarida, R., Galí, J., & Gertler, M. (1999). "The science of
  monetary policy: A New Keynesian perspective." *Journal of
  Economic Literature*.
- Clarida, R., Galí, J., & Gertler, M. (2000). "Monetary policy
  rules and macroeconomic stability." *Quarterly Journal of
  Economics*.
- Aron, J., & Muellbauer, J. (2007). "Review of monetary policy in
  South Africa during 1994-2004." *South African Journal of
  Economics*.
- Calvo, G., & Reinhart, C. (2002). "Fear of floating." *Quarterly
  Journal of Economics*.
- Newey, W., & West, K. (1987). "A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance
  matrix." *Econometrica*.
- Rey, H. (2013). "Dilemma not trilemma: The global financial cycle
  and monetary policy independence." *Jackson Hole Symposium*.
