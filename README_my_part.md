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

In the integrated rolling framework this is implemented in two versions:

- **Hard block liquidation**: immediately forces `Q -> 0` in one block trade at
  the liquidation timestamp. This answers the project stress question directly
  and may violate normal participation caps by design.
- **Capped residual liquidation**: respects the configured liquidation
  participation cap. If the position cannot be fully liquidated at the first
  timestamp, residual inventory remains active; liquidation trades have priority
  over new alpha trades until the residual is cleared. Residual inventory can be
  carried and is marked to market.

Run both integrated variants with:

```bash
python -m src.run_integrated_rolling_simulations --mode single_pair --pair-id 1 --liquidation-mode both
```

Integrated outputs include:

- `outputs/rolling_runs/pair_{pair_id}/stress/forced_liq_hard_block_trades.csv`
- `outputs/rolling_runs/pair_{pair_id}/stress/forced_liq_capped_residual_trades.csv`
- `outputs/rolling_runs/pair_{pair_id}/stress/forced_liq_hard_block_events.csv`
- `outputs/rolling_runs/pair_{pair_id}/stress/forced_liq_capped_residual_events.csv`
- fitted evaluator files for both liquidation paths.

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

## Integrated Use Of Teammate 2.1/2.2/2.3 Outputs

The integration layer reads teammate rolling pairs from:

```text
data/processed_2_1/rolling_full_universe/rolling_pair_summary.csv
```

For each selected pair, it loads the train/test processed bin parquet files,
generates my synthetic alpha on the test month, runs my OW strategy, then
evaluates the resulting trades under teammate fitted regressions.

Teammate fitted models are handled as regressions for within-bin price movement:

```text
ret_bps = 10000 * (midEnd - mid) / mid
```

`OW_transient` is interpreted as:

```text
ret_bps = intercept + b_flow x_flow + b_state ow_state_pre
```

where `ow_state_pre` is reproduced with teammate convention:

```text
state_pre_t = exp(-dt / half_life_sec) * state_post_{t-1}
state_post_t = state_pre_t + x_flow_t
```

`half_life_sec` is therefore a teammate regression time constant for feature
reproduction. It is not treated as strict half-life there. When using it inside
my OW strategy, it is converted to strict half-life as:

```text
H_minutes = half_life_sec * ln(2) / 60
```

`reduced_form` is interpreted as a fitted regression:

```text
ret_bps = intercept
        + b1 x_flow
        + b2 x_trade
        + b3 x_hidden
        + b4 x_flow_depth
        + b5 lobImb
        + b6 effLobImb
        + b7 spread_bps
```

It is not the course dynamic AFS simulator unless a separate structural model is
provided later.

Feature scaling reproduces teammate logic:

- `x_flow`, `x_trade`, and `x_hidden` are normalized by the train-month median
  daily absolute `orderFlow` scale.
- scales are computed from train month only and then applied to test month.
- no current test information is used to compute train scales.

Wrong-model stress uses the required convention:

```text
orderFlow_scenario = orderFlow_market + q_strategy
marginal_impact_bps = predicted_with_trade - predicted_market
```

Run a debug pair with:

```bash
python -m src.run_integrated_rolling_simulations --mode single_pair --pair-id 1 --save-trades
```

Run all selected pairs with:

```bash
python -m src.run_integrated_rolling_simulations --mode all_pairs --max-pairs 3 --save-trades
```

Outputs:

- `outputs/rolling_runs/pair_{pair_id}/alpha/`
- `outputs/rolling_runs/pair_{pair_id}/my_ow_trades.csv`
- `outputs/rolling_runs/pair_{pair_id}/fitted_proxy_strategy_trades.csv`
- `outputs/rolling_runs/pair_{pair_id}/ow_transient_regression_evaluator.csv`
- `outputs/rolling_runs/pair_{pair_id}/reduced_form_regression_evaluator.csv`
- `outputs/rolling_runs/pair_{pair_id}/stress/wrong_model_regression_evaluator.csv`
- `outputs/rolling_runs/all_pairs_strategy_summary.csv`
- `outputs/rolling_runs/all_pairs_fitted_proxy_summary.csv`
- `outputs/rolling_runs/all_pairs_wrong_model_summary.csv`
- `outputs/rolling_runs/integrated_rolling_report.md`

Important environment note: teammate processed market data are parquet files.
The Python environment must have `pyarrow` or `fastparquet` installed, or the
processed data must be exported to CSV before running the integration.

Important caveats:

- Do not call teammate `reduced_form` dynamic AFS.
- Do not silently map `x_flow` to structural lambda.
- `x_flow` can be used as a lambda proxy only with the explicit CLI flag
  `--use-x-flow-lambda-proxy`, and the outputs are labelled accordingly.
- Regression evaluator costs are model-implied diagnostics, not a full
  structural simulator.

### Fitted-Regression Proxy Strategy

The project asks for an optimal strategy under OW and the fitted model. The OW
case has a course-style closed form in target-impact space. Teammate's fitted
`reduced_form` model, however, is a linear regression for within-bin `ret_bps`,
not a structural impact dynamics. I therefore implement a fitted-model-aware
**local myopic quadratic-cost proxy** rather than claiming a dynamic closed-form
optimum.

The proxy uses the same scenario convention as the wrong-model stress:

```text
orderFlow_scenario = orderFlow_market + q_strategy
```

Because the regression is linear, the local marginal predicted return impact of
an additional strategy trade is:

```text
d ret_bps / dq =
    b_x_flow / flowScale
  + b_x_flow_depth / depth
  + b_x_trade / flowScale   # only if explicitly enabled
```

This is converted into a price impact slope:

```text
impact_slope_price_per_share = mid * (d ret_bps / dq) / 10000
```

The one-step proxy objective is:

```text
max_q q * alpha_price
      - impact_slope_price_per_share * q^2
      - inventory_penalty * (Q_prev + q)^2
```

with:

```text
alpha_price = mid * alpha_for_strategy
q_raw = (alpha_price - 2 * inventory_penalty * Q_prev)
        / (2 * impact_slope_price_per_share + 2 * inventory_penalty)
```

The proxy uses the same participation, trade, and inventory caps as the
integrated OW runner. Its output is saved to
`outputs/rolling_runs/pair_{pair_id}/fitted_proxy_strategy_trades.csv`, with
aggregate metrics in `outputs/rolling_runs/all_pairs_fitted_proxy_summary.csv`.

This is the honest fitted-regression strategy benchmark available from the
teammate model outputs. It should be labelled in the report as a local myopic
proxy induced by the fitted regression's marginal impact slope, not as the
course structural AFS optimum.

### Integrated PnL and Fitted-Cost Validation

The integrated rolling report now separates three concepts that should not be
confused:

1. **Gross alpha capture**: positive `gross_pnl` means the OW strategy captures
   the synthetic alpha before fitted impact costs.
2. **Internal OW normalized cost**: `signed_impact_cost_normalized` is useful for
   checking OW mechanics, but it is not on the same economic scale as price-unit
   fitted-regression costs.
3. **Fitted regression impact cost**: teammate regressions imply a marginal
   price move from our additional order flow.

For fitted regression costs, the convention is:

```text
orderFlow_scenario = orderFlow_market + q_strategy
marginal_impact_bps = pred_ret_bps_with_strategy - pred_ret_bps_market
marginal_impact_price = mid * marginal_impact_bps / 10000
fitted_cost = signed_volume * marginal_impact_price
net_pnl_fitted_model = gross_pnl - fitted_cost
```

A positive `fitted_cost` is adverse and is subtracted from gross PnL. Therefore,
negative fitted net PnL is not automatically a bug: it can occur when the
model-implied impact cost is larger than the alpha capture.

The validation report checks:

- `gross_pnl = position_before * delta_mid`
- `position_after = position_before + signed_volume`
- `orderFlow_scenario = orderFlow_market + signed_volume`
- `marginal_impact_bps = pred_with_trade - pred_market`
- fitted cost sign and bps-to-price conversion
- participation and position caps

Current report paths:

- `outputs/rolling_runs/integrated_rolling_report.md`
- `outputs/rolling_runs/integrated_validation_report.txt`
- `outputs/rolling_runs/integrated_validation_checks.csv`
- `outputs/rolling_runs/pair_{pair_id}/debug/debug_strategy_path_sample.csv`

### Figure Reproducibility

Integrated rolling figures are regenerated statelessly on each run. The runner:

- creates fresh matplotlib figures and closes them after saving;
- overwrites the latest convenience figures under `outputs/rolling_runs/figures/`
  and `outputs/rolling_runs/pair_{pair_id}/figures/`;
- cleans old latest PNG/PDF figures by default before plotting;
- records run metadata in `outputs/rolling_runs/latest_run_metadata.json`;
- archives each run's figures under `outputs/rolling_runs/runs/{run_id}/`.

The default CLI behavior is equivalent to:

```bash
python -m src.run_integrated_rolling_simulations --clean-output-figures
```

Use `--no-clean-output-figures` only when explicitly comparing existing latest
figures manually. Validation checks confirm required figures exist, are non-empty,
and were modified after the current run started.

### Experiment Management For Full OOS Runs

Integrated runs are now isolated under:

```text
outputs/full_runs/{experiment_name}/
```

Each experiment contains `metadata/`, `reports/`, `tables/`, and pair-level
folders with `reports/`, `figures/`, `tables/`, `trades/`, `stress/`, and
`debug/`. The pointer `outputs/rolling_runs/latest_experiment_path.txt` records
the latest experiment folder.

Recommended workflow:

```bash
python -m src.run_integrated_rolling_simulations \
  --mode single_pair \
  --pair-id 1 \
  --scenario baseline \
  --max-rows-per-pair 20000 \
  --save-trades \
  --experiment-name debug_pair1_baseline

python -m src.run_integrated_rolling_simulations \
  --mode single_pair \
  --pair-id 1 \
  --scenario baseline \
  --save-trades \
  --experiment-name full_pair1_baseline

python -m src.run_integrated_rolling_simulations \
  --mode single_pair \
  --pair-id 1 \
  --scenario signal_delay \
  --save-trades \
  --experiment-name full_pair1_signal_delay

python -m src.run_integrated_rolling_simulations \
  --mode single_pair \
  --pair-id 1 \
  --scenario wrong_model \
  --save-trades \
  --experiment-name full_pair1_wrong_model

python -m src.run_integrated_rolling_simulations \
  --mode single_pair \
  --pair-id 1 \
  --scenario forced_liquidation \
  --liquidation-mode both \
  --save-trades \
  --experiment-name full_pair1_forced_liq

python -m src.run_integrated_rolling_simulations \
  --mode single_pair \
  --pair-id 1 \
  --scenario sizing_sensitivity \
  --experiment-name full_pair1_sizing_sensitivity

python -m src.run_integrated_rolling_simulations \
  --mode single_pair \
  --pair-id 1 \
  --scenario all \
  --save-trades \
  --experiment-name full_pair1_all_scenarios

python -m src.run_integrated_rolling_simulations \
  --mode all_pairs \
  --scenario baseline \
  --experiment-name full_allpairs_baseline
```

When `--max-rows-per-pair` is omitted, the selected full out-of-sample test
month is processed. The report records `full_out_of_sample_run=True` and daily
Sharpe is treated as meaningful only when at least two dates are present.

### Strategy Implementation Audit

For baseline experiments, the implementation audit checks whether the OW
target-impact trades, fitted-regression evaluator, and fitted-regression proxy
strategy are mechanically consistent:

```bash
python -m src.strategy_implementation_audit \
  --experiment-path outputs/full_runs/full_pair1_baseline \
  --pair-id 1
```

It saves `audit_strategy_implementation_report.md`, CSV audit tables, and
`audit_figures/`. The audit compares OW and proxy turnover, cost bps of
notional turnover, proxy cost reconstruction, proxy internal costs versus the
full reduced-form evaluator on the same proxy trades, feature extrapolation,
order-flow unit consistency, timing, and fitted cost sign conventions.

To run it automatically after a baseline run:

```bash
python -m src.run_integrated_rolling_simulations \
  --mode single_pair \
  --pair-id 1 \
  --scenario baseline \
  --save-trades \
  --run-strategy-audit \
  --experiment-name full_pair1_baseline_audit
```

If `--save-trades` is omitted, strict OW row-level trade-file checks are marked
`SKIP`; evaluator-based sizing and cost diagnostics still run where possible.
