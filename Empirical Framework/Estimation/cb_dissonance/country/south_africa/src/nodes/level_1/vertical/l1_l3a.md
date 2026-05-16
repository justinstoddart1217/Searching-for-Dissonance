# `l1_l3a.py` — L1↔L3a Vertical Edge Test

Implements the three-function structure for the L1 → L3a vertical edge: a gate test, a consistency function, and a dissonance metric. Operates on the output of `SARBReactionFunction.fit()`. Stateless — all three functions take the `SARBReactionFunctionFit` object and derive everything from it.

The edge is vertical in the framework's typology: L1 (reaction function, rule-implied rate i*_t) and L3a (observed policy rate i_t) sit on adjacent levels. Dissonance here measures how far the bank's actual decision deviates from what the calibrated rule predicts.

## Public API

```python
from country.south_africa.src.nodes.level_1.vertical.l1_l3a import (
    correlation_test, consistency, dissonance
)

ct = correlation_test(fit)   # gate: is the rule admissible as a consistency model?
cn = consistency(fit)        # R(t) = i_t - i*_t
di = dissonance(fit, threshold_multiplier=2.0)  # D(t) = |R(t)|, flag where D > tau
```

All three return frozen dataclasses with a `metadata` dict.

## Three functions, one purpose

### 1. `correlation_test(fit) -> CorrelationTestResult`

The gate. Computes the bivariate relationship between observed i_t and rule-implied i*_t and tests it against the pre-registered admissibility bars.

Pre-registered (Chapter 3, framework spec):
- **R²(i_t, i*_t) ≥ 0.84** on calibration window
- **σ(ε) / σ(i_t) ≤ 0.40** on calibration window

Both must hold for the edge to be admissible. The OOS versions of both metrics are computed and reported but **do not gate** — they're diagnostic only. A rule that passes calibration but fails OOS still admits the edge for dissonance measurement; the OOS deterioration *is itself information* about regime change.

On calibration, bivariate R²(i_t, i*_t) is algebraically identical to the multivariate R² from the OLS fit (since i*_t is the OLS fitted value, the bivariate fit reproduces it exactly). On OOS, they differ — bivariate R² holds the calibrated coefficients fixed.

Returned fields:
- `r2_calib`, `r2_oos`: bivariate R² on each window
- `sigma_ratio_calib`, `sigma_ratio_oos`: σ(ε)/σ(i_t) on each window
- `passes_r2_calib`, `passes_sigma_ratio_calib`: pre-reg bar checks
- `passes_r2_oos`, `passes_sigma_ratio_oos`: diagnostic OOS checks
- `admissible`: `passes_r2_calib AND passes_sigma_ratio_calib`

### 2. `consistency(fit) -> ConsistencyResult`

Returns the signed residual series R(t) = i_t − i*_t. The framework specifies E[R] = 0 in the equilibrium regime; deviations from zero are the input to the dissonance metric.

The residual series spans the full sample (calibration + OOS, concatenated). No truncation, no transformation — this is just the raw residual from the OLS fit, presented under the framework's labelling convention.

Returned fields:
- `R`: pd.Series of signed residuals, indexed monthly 2005-01 to 2024-12
- `metadata`: sample mean (should be ~0 by OLS construction), sample SD, n_obs

### 3. `dissonance(fit, threshold_multiplier=2.0) -> DissonanceResult`

Returns D(t) = |R(t)| and flags observations where D(t) > τ.

Threshold definition: **τ = k × σ(ε_calib)** where k is the multiplier (default 2.0). σ(ε_calib) is the residual standard deviation on the calibration window only — the OOS residuals do not feed into the threshold, because that would let the test calibrate to its own test set.

The choice of k = 2 follows the 2-sigma convention. Under approximate Gaussian residuals it captures ~5% of months in normal regime as "false positives" — the cost of a tight enough threshold to detect genuine policy surprises.

Returned fields:
- `D`: pd.Series of |R(t)|
- `threshold_tau`: the numerical threshold τ in units of i_t (percent)
- `flagged`: pd.Series of bool, True where D > τ
- `metadata`: σ(ε_calib), τ, threshold multiplier, count and fraction flagged (total, calib-only, OOS-only)

## Composition

The three functions are designed to compose:

```python
fit = SARBReactionFunction().fit(data)
ct = correlation_test(fit)
if not ct.admissible:
    raise RuntimeError(f"L1->L3a edge rejected: R²={ct.r2_calib:.3f}, ratio={ct.sigma_ratio_calib:.3f}")
cn = consistency(fit)
di = dissonance(fit)
flagged = cn.R[di.flagged]  # signed residuals at flagged dates
```

The gate fails early. If admissible, R is the raw signal and D is the magnitude-with-threshold. The flagged dates are the dissonance episodes for L1↔L3a.

## Why not just read from `fit.diagnostics`?

The reaction function's diagnostic dict already contains R², σ(ε), σ(i_t). The correlation test could trivially read those. Two reasons it doesn't:

1. **Conceptual clarity.** The framework defines correlation test as a bivariate test between i_t and i*_t. Computing it explicitly from the residual series makes that algebra visible.
2. **OOS bivariate metrics.** The reaction function's diagnostics report multivariate-style σ_ratio for OOS (using prediction errors). The bivariate R² and σ_ratio on OOS are subtly different metrics — they're what the framework actually specifies for the gate.

On calibration the two coincide; on OOS they diverge slightly. Computing fresh is cheaper than explaining why we'd read one and re-compute the other.

## What's not covered here

- **Multi-edge composition.** When other edges are built (L1↔L2 horizontal, L3a↔L4 vertical, etc.) each gets its own module. The country-level composite D_c,t aggregates them, but lives in a separate node-aggregator module (not yet built).
- **Time-varying threshold.** τ is currently a scalar. A future refinement might use a rolling σ(ε) for regime-conditional thresholds. Defer.
- **Pre-registration audit trail.** The `metadata` dicts include the pre-reg bar values, but the full audit log (when bars were set, by whom, against what hash of the framework doc) lives separately.

## Failure modes

| Scenario | Behaviour |
|---|---|
| `fit` from a different country | Won't crash — `fit.residuals` and `fit.i_star` are pandas Series, no schema check. But the metadata edge will be wrong. Discipline: don't mix country `fit` objects with this module. |
| Calibration window has too few obs | Inherited from the upstream `fit`. If `fit.metadata['n_obs_calibration'] < 30`, the regression itself would have failed. |
| All residuals identical (degenerate) | σ(ε_calib) = 0 → τ = 0 → all non-zero D values flagged. Treat as data error in upstream. |
| Gate fails | `admissible=False`. The consistency and dissonance functions still run if called, but their output shouldn't be used for inference. The runner script halts after the gate. |
