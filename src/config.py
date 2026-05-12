"""Configuration objects for alpha generation, strategy, and stress tests."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AlphaConfig:
    """Configuration for synthetic alpha construction."""

    intraday_horizon_steps: int = 5
    intraday_strength: float = 1.0
    intraday_decay: float = 0.0
    overnight_strength: float = 1.0
    combined_intraday_weight: float = 0.7
    combined_overnight_weight: float = 0.3
    alpha_clip: Optional[float] = None


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for the simplified impact-aware trading strategy."""

    risk_aversion: float = 1.0
    max_position: float = 100_000.0
    max_trade_size: Optional[float] = 10_000.0
    liquidation_at_close: bool = True
    position_penalty: float = 0.01
    trade_penalty: float = 0.0
    target_position_scale: float = 1_000_000.0


@dataclass(frozen=True)
class StressTestConfig:
    """Configuration for strategy stress-test scenarios."""

    signal_delay_steps: int = 5
    forced_liquidation_time: Optional[str] = "12:00"
    wrong_model_assumed: str = "OW"
    wrong_model_true: str = "advanced"

