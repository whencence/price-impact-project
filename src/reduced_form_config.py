"""Configuration for the reduced-form AFS-style strategy."""

from dataclasses import dataclass
from math import log


@dataclass(frozen=True)
class ReducedFormStrategyConfig:
    """Configuration for the reduced-form fitted-model strategy placeholder."""

    alpha_col: str = "alpha_for_strategy"
    price_col: str = "mid"
    stock_col: str = "stock"
    date_col: str = "date"
    time_col: str = "time"
    timestamp_col: str = "timestamp"

    beta: float | None = None
    impact_half_life_minutes: float = 5.0
    lambda_base: float = 1.0

    volume_col: str = "trade"
    volume_window_minutes: float = 5.0
    market_volume_source: str = "trade"
    min_volume_state: float = 1e-12

    use_full_gamma_formula: bool = False
    use_slow_moving_liquidity_heuristic: bool = True

    mu_col: str | None = None
    mu_method: str = "backward_alpha_derivative"
    mu_horizon_minutes: float = 5.0
    winsorize_mu_quantile: float | None = 0.995

    scaling_factors_path: str | None = "outputs/scaling/scaling_factors_20d.csv"
    use_scaling_factors: bool = True
    fallback_sigma: float | None = None
    fallback_ADV: float | None = None

    max_abs_position: float | None = None
    max_abs_trade: float | None = None
    max_participation_rate: float | None = 0.05
    liquidate_at_close: bool = True

    skip_missing_scaling: bool = True
    initial_position: float = 0.0
    initial_impact: float = 0.0
    translation_method: str = "position_formula"

    @property
    def effective_beta(self) -> float:
        """Return configured beta, or beta implied by impact half-life."""

        return float(self.beta if self.beta is not None else log(2.0) / self.impact_half_life_minutes)

    def __post_init__(self) -> None:
        """Validate configuration values."""

        for name in ["alpha_col", "price_col", "stock_col", "date_col", "time_col", "timestamp_col", "volume_col"]:
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.beta is not None and self.beta <= 0:
            raise ValueError("beta must be None or positive")
        if self.impact_half_life_minutes <= 0:
            raise ValueError("impact_half_life_minutes must be positive")
        if self.lambda_base <= 0:
            raise ValueError("lambda_base must be positive")
        if self.volume_window_minutes <= 0:
            raise ValueError("volume_window_minutes must be positive")
        if self.min_volume_state <= 0:
            raise ValueError("min_volume_state must be positive")
        if self.use_full_gamma_formula and self.use_slow_moving_liquidity_heuristic:
            # Full formula takes priority in the strategy implementation.
            pass
        if self.mu_method not in {"backward_alpha_derivative", "alpha_over_horizon", "provided_column"}:
            raise ValueError("mu_method must be backward_alpha_derivative, alpha_over_horizon, or provided_column")
        if self.mu_method == "provided_column" and not self.mu_col:
            raise ValueError("mu_col must be provided when mu_method='provided_column'")
        if self.mu_horizon_minutes <= 0:
            raise ValueError("mu_horizon_minutes must be positive")
        if self.winsorize_mu_quantile is not None and not 0.5 < self.winsorize_mu_quantile < 1.0:
            raise ValueError("winsorize_mu_quantile must be None or in (0.5, 1.0)")
        if self.fallback_sigma is not None and self.fallback_sigma <= 0:
            raise ValueError("fallback_sigma must be None or positive")
        if self.fallback_ADV is not None and self.fallback_ADV <= 0:
            raise ValueError("fallback_ADV must be None or positive")
        if self.max_abs_position is not None and self.max_abs_position <= 0:
            raise ValueError("max_abs_position must be None or positive")
        if self.max_abs_trade is not None and self.max_abs_trade <= 0:
            raise ValueError("max_abs_trade must be None or positive")
        if self.max_participation_rate is not None and not 0 < self.max_participation_rate < 1:
            raise ValueError("max_participation_rate must be None or in (0, 1)")
        if self.translation_method not in {"position_formula", "inverse_sde"}:
            raise ValueError("translation_method must be position_formula or inverse_sde")
