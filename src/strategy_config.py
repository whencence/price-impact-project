"""Configuration for OW optimal strategy experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OWStrategyConfig:
    """Configuration for the placeholder OW strategy implementation."""

    alpha_col: str = "alpha_for_strategy"
    price_col: str = "mid"
    stock_col: str = "stock"
    date_col: str = "date"
    time_col: str = "time"
    timestamp_col: str = "timestamp"

    scaling_factors_path: str | None = "outputs/scaling/scaling_factors_20d.csv"
    use_scaling_factors: bool = True
    fallback_sigma: float | None = None
    fallback_ADV: float | None = None
    impact_model_type: str = "linear"

    impact_lambda: float = 1.0
    impact_half_life_minutes: float = 5.0

    max_abs_position: float | None = None
    max_abs_trade: float | None = None

    liquidate_at_close: bool = True
    liquidation_penalty: float = 0.0

    winsorize_alpha_dot_quantile: float | None = 0.995
    max_abs_target_impact: float | None = None

    initial_position: float = 0.0
    initial_impact: float = 0.0

    def __post_init__(self) -> None:
        """Validate strategy configuration values."""

        if self.impact_model_type not in {"linear", "sqrt"}:
            raise ValueError("impact_model_type must be 'linear' or 'sqrt'")
        if not self.use_scaling_factors:
            if self.fallback_sigma is None or self.fallback_sigma <= 0:
                raise ValueError("fallback_sigma must be positive when use_scaling_factors=False")
            if self.fallback_ADV is None or self.fallback_ADV <= 0:
                raise ValueError("fallback_ADV must be positive when use_scaling_factors=False")
        if self.fallback_sigma is not None and self.fallback_sigma <= 0:
            raise ValueError("fallback_sigma must be None or positive")
        if self.fallback_ADV is not None and self.fallback_ADV <= 0:
            raise ValueError("fallback_ADV must be None or positive")
        if self.impact_lambda <= 0:
            raise ValueError("impact_lambda must be positive")
        if self.impact_half_life_minutes <= 0:
            raise ValueError("impact_half_life_minutes must be positive")
        if self.max_abs_position is not None and self.max_abs_position <= 0:
            raise ValueError("max_abs_position must be None or positive")
        if self.max_abs_trade is not None and self.max_abs_trade <= 0:
            raise ValueError("max_abs_trade must be None or positive")
        if self.winsorize_alpha_dot_quantile is not None:
            if not 0.5 < self.winsorize_alpha_dot_quantile < 1.0:
                raise ValueError("winsorize_alpha_dot_quantile must be None or in (0.5, 1.0)")
        for name in ["alpha_col", "price_col", "stock_col", "date_col", "time_col", "timestamp_col"]:
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
