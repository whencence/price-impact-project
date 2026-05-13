"""Configuration objects for alpha generation, strategy, and stress tests."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AlphaConfig:
    """Configuration for homework synthetic alpha construction."""

    horizon_minutes: float = 5.0
    target_corr: float = 0.10
    random_seed: int = 42
    alpha_clip: float | None = None
    price_col: str = "mid"
    stock_col: str = "stock"
    date_col: str = "date"
    time_col: str = "time"
    timestamp_col: str = "timestamp"
    future_tolerance_seconds: float = 30.0
    alpha_decay_half_life_minutes: float = 5.0
    alpha_decay_half_life_grid_minutes: tuple[float, ...] = (1.0, 5.0, 30.0, 60.0)
    group_estimation: str = "global"

    def __post_init__(self) -> None:
        """Validate alpha configuration values."""

        if self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        if not 0.0 < self.target_corr < 1.0:
            raise ValueError("target_corr must be strictly between 0 and 1")
        if self.future_tolerance_seconds < 0:
            raise ValueError("future_tolerance_seconds must be non-negative")
        if self.alpha_decay_half_life_minutes <= 0:
            raise ValueError("alpha_decay_half_life_minutes must be positive")
        if any(value <= 0 for value in self.alpha_decay_half_life_grid_minutes):
            raise ValueError("all alpha decay half-lives must be positive")
        valid_groups = {"global", "by_stock"}
        if self.group_estimation not in valid_groups:
            raise ValueError(f"group_estimation must be one of {sorted(valid_groups)}")
        for name in ["price_col", "stock_col", "date_col", "time_col", "timestamp_col"]:
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for the simplified impact-aware trading strategy."""

    risk_aversion: float = 1.0
    max_position: float = 1_000.0
    max_trade_size: float | None = None
    liquidation_at_close: bool = True
    position_penalty: float = 0.0
    trade_penalty: float = 0.0
    target_position_scale: float = 1.0


@dataclass(frozen=True)
class StressTestConfig:
    """Configuration for strategy stress-test scenarios."""

    signal_delay_steps: int = 1
    forced_liquidation_time: str | None = "12:00"
    wrong_model_assumed: str = "OW"
    wrong_model_true: str = "advanced"
