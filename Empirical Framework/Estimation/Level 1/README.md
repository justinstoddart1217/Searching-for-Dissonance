# Central Bank Dissonance — Empirical Framework

Empirical implementation of the structural dissonance framework specified in
`thesis_structure.pdf` and `central_bank_dissonance_proposal.pdf`. This codebase
operationalises the three core measurement objects:

1. **Intensity** — is the relationship structurally tight enough for the residual
   to be diagnostically informative?
2. **Consistency function `Rⁿ`** — maps a node's components to zero in the
   equilibrium regime.
3. **Dissonance function `Dⁿ = |Rⁿ|`** — magnitude of departure from equilibrium,
   compared against a country-specific threshold `τⁿ`.

Architecture follows the framework taxonomy: levels are the primary organising
axis; within each level the edges are partitioned into *intra-node*, *vertical*,
and *inter-node* (peer) relationships.

## Directory layout

```
cb_dissonance/
├── README.md                        # this file
├── config/
│   └── preregistration.yaml         # country-by-country specifications (pre-registered)
├── src/
│   ├── data/
│   │   └── schema.py                # canonical input data schema
│   ├── level_1/
│   │   ├── intra_node/              # rule-coefficient drift (Var(θ̂_t))
│   │   │   ├── reaction_function.py # static θ̂ estimation (OLS, GMM)
│   │   │   ├── rolling_estimation.py# rolling-window θ̂_t (transparent comparator)
│   │   │   ├── tvp_estimation.py    # Kalman-filter TVP θ̂_t (headline)
│   │   │   └── diagnostics.py       # intensity, consistency, dissonance, thresholds
│   │   ├── vertical/                # L1↔L3a, L1↔L3b, … (downstream — uses same fit)
│   │   └── inter_node/              # n/a at L1 (single node)
│   ├── level_3/                     # L3a–L3f intra; L3↔L3 peer
│   ├── composite/                   # aggregation D_{c,t}
│   └── …
├── notebooks/
│   └── 01_l1_intra_pilot.py         # runnable end-to-end pilot on synthetic data
└── tests/
    └── test_l1_intra.py             # unit tests using synthetic ground truth
```

## L1 intra-node — what this module measures

The reaction function at time `t`:

```
i*_t = ρ_t · i_{t-1} + (1 − ρ_t) · [r* + α_t·(E_t π_{t+h} − π*) + β_t·E_t ỹ_{t+k} + γ_t·Δq_t + δ_t·i^foreign_t]
```

L1 **intra-node** dissonance asks whether `θ_t = (α_t, β_t, ρ_t, γ_t, δ_t)` is
stable across `t`. The rule changing under the bank's feet is a silent regime
shift even when no individual decision looks unusual.

L1 ↔ L3a (vertical, downstream) asks whether the bank executed the rule at any
given `t`: `ε_t = i_t − i*_t(θ̂_t)`. The same TVP fit produces both metrics.

## Methodology stack

Three estimators, all sharing one specification (`ReactionFunctionSpec`):

| Estimator | Role | Output |
|---|---|---|
| `StaticReactionFunction` | calibration baseline `θ̂_baseline` | scalar coefficient vector |
| `RollingReactionFunction` | transparent comparator (no state-space assumptions) | `θ̂_t` from k-window OLS |
| `TVPReactionFunction` | headline drift metric | filtered & smoothed `θ̂_t`, innovations `η_t` |

The headline intra-L1 dissonance uses the TVP innovations. The rolling-window
comparator is reported alongside as a robustness check that the drift signal
is not an artefact of the state-space prior on `Q`.

## Intensity gating

Before computing `D^{L1_intra}`, the reaction function must clear the intensity
sufficiency condition:

- R² of the fitted rule on `i_t` > 0.7
- σ(ε)/σ(i) < 0.4 on the calibration window
- Granger causality from `{π gap, ỹ, Δq, i^foreign}` → `i_t` at p < 0.05

Failing intensity → the country is flagged low-intensity for L1 and downweighted
in the composite. This is a pre-registered rule, not a discretionary call.

## Data interface

Estimation code accepts a `pandas.DataFrame` conforming to `src/data/schema.py`.
The data layer (loaders, frequency harmonisation, real-time data handling) is
deliberately separated from econometrics — Bloomberg/Refinitiv/IMF/BIS sourcing
plugs in below the schema boundary.

## Pre-registration discipline

Country-specific specifications — which regressors, which horizons, which
calibration windows — live in `config/preregistration.yaml`, not in code.
This makes the spec hash-stable, timestamp-able, and auditable for the
defence.
