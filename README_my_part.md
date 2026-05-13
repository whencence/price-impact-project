# My Project Modules: Synthetic Alpha, Strategy, Stress Tests

This module implements my parts of the MSc project:

- 2.4 Synthetic Alphas
- 2.5 Optimal Trading Strategy
- 2.7 Sensitivity Analysis and Stress Testing

The code is modular because my teammate owns the final data preparation, rolling-window calibration, impact model fitting, backtest engine, and polished performance visualization.

## Homework Synthetic Alpha

For now, only the intraday homework alpha is implemented. There is no overnight alpha and no combined alpha.

The alpha is:

```text
alpha_t^h = x r_t^h + y DeltaW_t^h / P_t
```

where:

- `P_t` is the observed midprice
- `r_t^h = (P_{t+h} - P_t) / P_t`
- `DeltaW_t^h ~ Normal(0, h)` is independent Brownian noise
- `x` controls the signal component
- `y` controls the noise component

To target correlation `rho`:

```text
x = rho^2
y = rho sqrt(1-rho^2) sqrt( Var(r_t^h) / (E[1/P_t^2] h) )
```

`Var(r_t^h)` and `E[1/P_t^2]` are estimated from the available market data. The current production path uses global moment estimation across valid bin rows.

## Section 2.4 Synthetic Alpha

The production alpha pipeline uses `binSamples` only. It uses `P_t = mid_t`; `midEnd` is kept only as a diagnostic column where available. Forecast horizons are clock-time horizons in minutes, not row counts. The old row-based `h=5` convention was approximately 50 seconds when bins were spaced by 10 seconds, so it is not the final project convention.

Future returns use clock-time matching within each `stock` and `date`: for a timestamp `t`, the code finds the first available observation with timestamp `>= t + h` and rejects it if it is outside the tolerance. This avoids crossing stocks or trading days.

Alpha decay also uses clock time. The decayed state is:

```text
A_t = phi_t A_{t-1} + (1 - phi_t) alpha_t
phi_t = exp(-ln(2) dt_minutes / H)
```

where `H` is the alpha decay half-life in minutes and `dt_minutes` is the actual timestamp gap.

The baseline scenario is:

- `h = 5 minutes`
- `rho = 0.10`
- `H = 5 minutes`

The sensitivity grid is:

- `h in [1, 5, 10]` minutes
- `rho in [0.05, 0.10, 0.20, 0.30, 0.50]`
- `H in [1, 5, 30, 60]` minutes

`rho` and `H` are sensitivity axes. They should not be selected by maximizing in-sample PnL.

Run from a notebook or script:

```python
from src.alpha import run_synthetic_alpha_grid_clock_time

alpha_outputs, diagnostics_df = run_synthetic_alpha_grid_clock_time(
    bin_df,
    horizons_minutes=[1.0, 5.0, 10.0],
    rhos=[0.05, 0.10, 0.20, 0.30, 0.50],
    baseline_horizon_minutes=5.0,
    baseline_rho=0.10,
    decay_half_lives_minutes=[1.0, 5.0, 30.0, 60.0],
    baseline_decay_half_life_minutes=5.0,
    output_dir=PROJECT_ROOT / "outputs" / "alphas",
    random_seed=42,
)
```

Outputs:

- `outputs/alphas/synthetic_alpha_baseline_h5m_rho010.csv`
- `outputs/alphas/synthetic_alpha_baseline_h5_rho010.csv` compatibility copy; `h5m` is the correct clock-time name
- `outputs/alphas/synthetic_alpha_diagnostics.csv`
- `outputs/alphas/synthetic_alpha_decay_diagnostics.csv`
- `outputs/alphas/synthetic_alpha_decay_metadata.csv`
- `outputs/alphas/figures/`

Create report-ready validation tables and figures with:

```bash
python -m src.alpha_reporting
```

Additional reporting outputs:

- `outputs/alphas/synthetic_alpha_report_table.csv`
- `outputs/alphas/synthetic_alpha_validation_summary.txt`
- clean validation figures under `outputs/alphas/figures/`

## Assumptions

- binSamples data arrives as a pandas DataFrame with at least `date`, `time`, `stock`, and `mid`.
- Optional columns such as `spread`, `ret_1m`, and `window_id` are preserved when present.
- Impact model parameters are currently passed as dictionaries.
- Strategy PnL is an approximate fallback implementation for development only.

TODO: replace the approximate PnL calculation with the teammate's Waelbroeck/backtest simulator once available.

## Expected Market Data Schema

Required columns:

- `date`: date or string
- `time`: intraday time
- `stock`: string
- `mid`: float

Optional columns:

- `spread`: float
- `ret_1m`: float
- `window_id`: rolling-window identifier

## Expected Impact Parameter Format

```python
impact_params = {
    "OW": {
        "AAPL": {"lambda": 1e-6, "rho": 0.1},
        "MSFT": {"lambda": 1.2e-6, "rho": 0.12},
        "__universal__": {"lambda": 1e-6, "rho": 0.1},
    },
    "advanced": {
        "AAPL": {"lambda": 0.8e-6, "rho": 0.08, "eta": 0.5},
        "MSFT": {"lambda": 1.1e-6, "rho": 0.10, "eta": 0.6},
        "__universal__": {"lambda": 0.8e-6, "rho": 0.08, "eta": 0.5},
    },
}
```

## Run The Demo

From the project root:

```bash
python -m src.run_my_part_demo
```

The demo writes:

- `outputs/my_part_demo/alphas.csv`
- `outputs/my_part_demo/trades_baseline.csv`
- `outputs/my_part_demo/stress_comparison.csv`

## Integration Plan

1. Replace `src.mock_data.make_mock_market_data` with the teammate's prepared data output.
2. Preserve `window_id` for rolling-window compatibility.
3. Replace `src.mock_data.make_mock_impact_params` with fitted impact parameters.
4. Keep using `src.alpha.build_synthetic_alpha` for the homework alpha.
5. Use `src.strategy.generate_trades_for_alpha` for target/trade logic, or pass its output into the final backtest engine.
6. Use `src.stress_tests` around the final simulator for delayed signal, wrong impact model, and forced liquidation tests.

## Section 2.5 OW Optimal Trading Strategy

The OW strategy uses:

```text
outputs/alphas/strategy_alpha_input_h5m_rho010_H5m.csv
```

with `alpha_t = alpha_for_strategy`. The alpha derivative is a backward
difference in clock-time minutes within each stock/date group:

```text
alpha_dot_t = (alpha_t - alpha_{t-1}) / dt_minutes
```

The deterministic-alpha OW target is:

```text
I_target = 0.5 alpha - beta_I^{-1} alpha_dot
beta_I = ln(2) / H_I
```

The discrete OW impact mechanics are:

```text
I_before = exp(-beta_I dt) I_prev
required_qtilde = (I_target - I_before) / lambda
```

The course OW model drives impact with normalized signed volume:

```text
I_{t+dt} - I_t = -beta_I I_t dt + lambda sigma q_t / ADV
qtilde = sigma q_t / ADV
```

Therefore the implementation distinguishes:

- `signed_volume`: the real signed traded volume `q_t`
- `normalized_trade`: `qtilde = sigma * signed_volume / ADV`
- `trade`: a backward-compatible alias for `signed_volume`

For the linear OW baseline:

```text
signed_volume = required_qtilde * ADV / sigma
normalized_trade = sigma * signed_volume / ADV
I_after = I_before + lambda normalized_trade
Q_new = Q_old + signed_volume
```

For the optional square-root variant:

```text
qtilde = sigma sign(q_t) sqrt(abs(q_t) / ADV)
```

Baseline remains the linear OW model unless explicitly changed.

`sigma` is the 20-day trailing average of daily intraday price volatility, and
`ADV` is the 20-day trailing average of daily absolute traded volume. Both are
computed stock-by-stock from `binSamples` using previous trading days only, so
the current day is excluded.

Wealth is computed as:

```text
gross_pnl = Q_prev DeltaS
quadratic_impact_cost_normalized = 0.5 lambda normalized_trade^2
signed_impact_cost_normalized = I_before normalized_trade + quadratic_impact_cost_normalized
net_pnl = gross_pnl - signed_impact_cost_normalized
```

These cost columns are normalized diagnostics until the final calibrated cost
scale is agreed with the fitted model/backtest module.

Run the baseline OW mechanics with:

```bash
python -m src.run_ow_strategy
```

Outputs are saved under `outputs/strategy/ow/`, including:

- `ow_strategy_trades.csv`
- `ow_daily_metrics.csv`
- `ow_summary_metrics.csv`
- `ow_strategy_validation_report.txt`
- figures under `outputs/strategy/ow/figures/`

Important caveat: `impact_lambda = 1.0` is currently a placeholder. The code
validates strategy mechanics, but the economic scale of trades, costs, and PnL
is not final until calibrated OW parameters arrive from the fitted model module.
Once available, replace the placeholder with stock-specific calibrated lambdas
through the same interface.

`H_I` remains a clock-time half-life in minutes. The sensitivity grid is
`H_I in [1, 5, 30, 60]` minutes.

## Section 2.5 Reduced-Form AFS Strategy

My teammate described the enhanced fitted model as a reduced form of AFS. The
exact fitted parameter output is not available yet, so the implementation is
parameterized and uses placeholder defaults.

The reduced-form liquidity signal is:

```text
v_t = rolling sum of observed absolute market volume
lambda_t = lambda_base / sqrt(v_t)
```

The local volume state `v_t` is computed from `binSamples` market `trade` over a
rolling clock-time window, using actual timestamps and resetting by stock/date.

The baseline target impact uses the slow-moving liquidity heuristic:

```text
I*_t = 0.5 alpha_t - beta^{-1} mu_t
```

where `alpha_t = alpha_for_strategy`, and the default `mu_t` is a causal
backward derivative of alpha in minutes. The full gamma formula is implemented
as an option:

```text
I*_t = ((beta + gamma'_t) / (2 beta + gamma'_t)) alpha_t
       - (1 / (2 beta + gamma'_t)) mu_t
```

The default trade translation uses the course position formula because it is
more robust to small target-impact errors:

```text
Q_t = I_t / lambda_t + sum_{s <= t} beta I_s dt / lambda_s
```

An inverse-SDE translation method is also exposed for diagnostics:

```text
Delta Q_t = (beta I_t dt + Delta I_t) / lambda_t
```

Run the reduced-form strategy with:

```bash
python -m src.run_reduced_form_strategy
```

Outputs are saved under `outputs/strategy/reduced_form/`:

- `reduced_form_strategy_trades.csv`
- `reduced_form_daily_metrics.csv`
- `reduced_form_summary_metrics.csv`
- `reduced_form_validation_report.txt`
- `reduced_form_sensitivity_summary.csv` from the notebook
- figures under `outputs/strategy/reduced_form/figures/`

Important caveats:

- `lambda_base = 1.0` is a placeholder until fitted parameters arrive.
- `beta` defaults to `ln(2) / impact_half_life_minutes`.
- Stock-specific fitted parameters can be added later through the config/loaders.
- This is a reduced-form AFS-style strategy, not the full nonlinear AFS model.
- The baseline uses the slow-moving liquidity heuristic; full gamma is optional.
- Outputs are mechanical until final calibrated parameters are plugged in.

The module exposes `run_reduced_form_strategy(df, config)`, metrics, and
validation functions so section 2.7 stress tests can wrap delayed signal,
forced liquidation, and wrong-model scenarios around the same interface.

## Section 2.7 Sensitivity Analysis and Stress Testing

The stress framework evaluates the main degrees of freedom in the project:

- alpha strength `rho`
- alpha forecast horizon `h`
- alpha decay half-life `H_alpha`
- impact model choice
- impact lambda
- impact half-life `H_I`

The OW implementation is fully supported. Reduced-form/fitted-model hooks are
kept explicit, but final fitted-model stress results should wait until the
calibrated parameters are available.

Sensitivity grids:

- `rho in [0.05, 0.10, 0.20, 0.30, 0.50]`
- `h in [1, 5, 10]` minutes, when scenario alpha inputs exist
- `H_alpha in [1, 5, 30, 60]` minutes, when alpha state columns exist
- impact lambda multiplier in `[0.5, 1.0, 2.0]`
- `H_I in [1, 5, 30, 60]` minutes

Stress tests:

1. Signal delayed by one minute:

```text
alpha_delayed(t) = last alpha at or before t - 1 minute
```

2. Forced liquidation at 12:00:

```text
block trade = -current position
```

The default convention stops trading after the forced liquidation for that
stock/day.

3. Wrong impact parameters:

```text
generate trades under assumed OW parameters
recompute wealth with the same trades under true OW parameters
```

This keeps the trade path fixed and isolates the parameter misspecification
effect.

Run the full section 2.7 framework with:

```bash
python -m src.run_stress_tests
```

Outputs:

- `outputs/stress/sensitivity_summary.csv`
- `outputs/stress/stress_summary.csv`
- `outputs/stress/all_scenarios_summary.csv`
- `outputs/stress/stress_validation_report.txt`
- scenario trades such as `signal_delay_trades.csv`, `forced_liquidation_trades.csv`, and `wrong_impact_trades.csv`
- figures under `outputs/stress/figures/`

Caveats:

- Until fitted reduced-form parameters are finalized, wrong-impact-model stress
  is represented by wrong OW parameter stress rather than fake fitted-model
  results.
- If `lambda` is still a placeholder, stress outputs validate framework
  mechanics but are not final economic conclusions.
- Final report results should be rerun on the selected out-of-sample period with
  calibrated in-sample parameters and enough trailing data for sigma/ADV.
