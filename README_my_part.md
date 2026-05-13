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
