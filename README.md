# My Project Modules: Synthetic Alphas, Strategy, Stress Tests

This code implements my parts of the MSc project:

- 2.4 Synthetic Alphas
- 2.5 Optimal Trading Strategy
- 2.7 Sensitivity Analysis and Stress Testing

It is intentionally modular because the final data preparation, rolling-window calibration, impact model fitting, backtest engine, and polished visualization modules are owned by my teammate.

## Assumptions

- Market data arrives as a pandas DataFrame with at least `timestamp`, `date`, `ticker`, `mid`, and `volume`.
- Optional columns such as `spread`, `ret_1m`, `overnight_ret`, and `window_id` are preserved when present.
- Impact model parameters are currently passed as dictionaries. Ticker-specific parameters are used when available, otherwise `__universal__` is used.
- Strategy PnL is an approximate fallback implementation for development only.

TODO: replace the approximate PnL path with the teammate's Waelbroeck/backtest simulator once available.

## Expected Market Data Schema

Required columns:

- `timestamp`: pandas datetime
- `date`: date or string
- `ticker`: string
- `mid`: float
- `volume`: float

Optional columns:

- `spread`: float
- `ret_1m`: float
- `overnight_ret`: float
- `window_id`: rolling-window identifier

## Expected Impact Parameter Format

```python
impact_params = {
    "OW": {
        "AAPL": {"lambda": 1e-6, "rho": 0.1},
        "__universal__": {"lambda": 1e-6, "rho": 0.1},
    },
    "advanced": {
        "AAPL": {"lambda": 0.8e-6, "rho": 0.08, "eta": 0.5},
        "__universal__": {"lambda": 0.8e-6, "rho": 0.08, "eta": 0.5},
    },
}
```

## How To Run The Demo

From the project root:

```bash
python -m src.run_my_part_demo
```

The demo writes CSV files to:

```text
outputs/my_part_demo/
```

Generated files:

- `alphas.csv`
- `trades_intraday.csv`
- `trades_overnight.csv`
- `trades_combined.csv`
- `stress_comparison.csv`

## Integration Plan

1. Replace `src.mock_data.make_mock_market_data` with the teammate's prepared market-data output.
2. Pass rolling-window identifiers through `window_id`; these modules preserve the column but do not perform rolling calibration.
3. Replace `src.mock_data.make_mock_impact_params` with fitted parameters from the teammate's impact-model module.
4. Keep using `src.alpha` to build synthetic alpha columns.
5. Use `src.strategy.generate_trades_for_alpha` for target/trade logic, or pass its outputs into the final backtest engine.
6. Use `src.stress_tests` to run delayed-signal, wrong-model, and forced-liquidation scenarios around the final simulator.

