"""Build synthetic alpha inputs on teammate processed test-month data."""

from pathlib import Path

import pandas as pd

from src.alpha import (
    add_alpha_decay_state_clock_time,
    build_synthetic_alpha_clock_time,
)
from src.config import AlphaConfig


def build_alpha_input_for_pair(
    test_raw_df: pd.DataFrame,
    pair_id: int,
    test_month: str,
    stocks: list[str],
    alpha_horizon_minutes: float = 5.0,
    rho: float = 0.10,
    alpha_decay_half_life_minutes: float = 5.0,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Generate baseline synthetic alpha on a rolling-pair test month."""

    data = test_raw_df[test_raw_df["stock"].astype(str).isin(stocks)].copy()
    data["timestamp"] = pd.to_datetime(data["datetime"], errors="coerce")
    data["date"] = data["trading_date"].astype(str)
    cfg = AlphaConfig(
        horizon_minutes=alpha_horizon_minutes,
        target_corr=rho,
        price_col="mid",
        stock_col="stock",
        date_col="date",
        time_col="time",
        timestamp_col="timestamp",
        alpha_decay_half_life_minutes=alpha_decay_half_life_minutes,
    )
    alpha_df = build_synthetic_alpha_clock_time(data, cfg)
    alpha_df = add_alpha_decay_state_clock_time(
        alpha_df,
        alpha_col="alpha_synthetic",
        half_life_minutes=alpha_decay_half_life_minutes,
        config=cfg,
        output_col=f"alpha_state_H{int(alpha_decay_half_life_minutes)}m",
    )
    alpha_col = f"alpha_state_H{int(alpha_decay_half_life_minutes)}m"
    alpha_df["alpha_raw"] = alpha_df["alpha_synthetic"]
    alpha_df["alpha_for_strategy"] = alpha_df[alpha_col]
    alpha_df["pair_id"] = pair_id
    alpha_df["test_month"] = str(test_month)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rho_key = f"{int(round(rho * 100)):03d}"
        path = out_dir / f"strategy_alpha_input_pair_{pair_id}_h{alpha_horizon_minutes:g}m_rho{rho_key}_H{alpha_decay_half_life_minutes:g}m.csv"
        alpha_df.to_csv(path, index=False)
    return alpha_df
