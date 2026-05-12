"""Mock data generators for local development before integration."""

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd


def make_mock_market_data(
    n_tickers: int = 3,
    n_days: int = 5,
    minutes_per_day: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic mock intraday market data.

    Prices follow a simple geometric random walk with plausible intraday
    timestamps. window_id is included for rolling-window compatibility.
    """

    if n_tickers < 1 or n_days < 1 or minutes_per_day < 2:
        raise ValueError("n_tickers and n_days must be positive; minutes_per_day must be >= 2")

    rng = np.random.default_rng(seed)
    tickers = [f"STK{i + 1}" for i in range(n_tickers)]
    start_day = date(2024, 1, 2)
    rows = []
    for ticker_idx, ticker in enumerate(tickers):
        price = 100.0 + 10.0 * ticker_idx
        for day_idx in range(n_days):
            current_date = start_day + timedelta(days=day_idx)
            start_dt = datetime.combine(current_date, time(9, 30))
            overnight_jump = rng.normal(0.0002, 0.003)
            price *= float(np.exp(overnight_jump))
            for minute in range(minutes_per_day):
                ts = start_dt + timedelta(minutes=minute)
                ret = rng.normal(0.0, 0.001)
                if minute == 0:
                    ret_for_column = 0.0
                else:
                    price *= float(np.exp(ret))
                    ret_for_column = ret
                rows.append(
                    {
                        "timestamp": pd.Timestamp(ts),
                        "date": current_date.isoformat(),
                        "ticker": ticker,
                        "mid": price,
                        "volume": float(rng.integers(1_000, 10_000)),
                        "spread": float(0.01 + rng.random() * 0.03),
                        "ret_1m": float(ret_for_column),
                        "window_id": f"window_{day_idx // 2}",
                    }
                )

    return pd.DataFrame(rows).sort_values(["ticker", "date", "timestamp"]).reset_index(drop=True)


def make_mock_impact_params(tickers: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    """Create deterministic mock impact parameters for given tickers."""

    params: dict[str, dict[str, dict[str, float]]] = {
        "OW": {"__universal__": {"lambda": 1e-6, "rho": 0.10}},
        "advanced": {"__universal__": {"lambda": 0.8e-6, "rho": 0.08, "eta": 0.50}},
    }
    for idx, ticker in enumerate(tickers):
        params["OW"][ticker] = {"lambda": 1e-6 * (1.0 + 0.15 * idx), "rho": 0.10 + 0.01 * idx}
        params["advanced"][ticker] = {
            "lambda": 0.8e-6 * (1.0 + 0.20 * idx),
            "rho": 0.08 + 0.01 * idx,
            "eta": 0.50 + 0.05 * idx,
        }
    return params

