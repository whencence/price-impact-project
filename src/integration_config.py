"""Configuration for integrating teammate processed data and fitted regressions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TeammateDataConfig:
    """Paths to teammate processed data and fitted outputs."""

    processed_21_dir: str = "data/processed_2_1"
    processed_22_dir: str = "data/processed_2_2_rolling_baseline"
    rolling_pair_summary_path: str = "data/processed_2_1/rolling_full_universe/rolling_pair_summary.csv"
    baseline_20stocks_summary_path: str = "data/processed_2_1/baseline_20stocks/baseline_20stocks_summary.csv"
    ow_params_path: str = "data/processed_2_2_rolling_baseline/parameters/ow_transient_params_by_pair_stock.csv"
    reduced_form_params_path: str = "data/processed_2_2_rolling_baseline/parameters/reduced_form_params_by_pair_stock.csv"
    monthly_bin_dir: str = "data/processed_2_1/monthly_standardised/bin"
    monthly_fill_dir: str = "data/processed_2_1/monthly_standardised/fills"


@dataclass(frozen=True)
class IntegratedRunConfig:
    """Configuration for integrated rolling simulations."""

    output_dir: str = "outputs/integration"
    rolling_output_dir: str = "outputs/rolling_runs"
    run_mode: str = "single_pair"
    debug_pair_id: int | None = 1
    selected_pair_ids: tuple[int, ...] | None = None
    use_baseline_20stocks: bool = False
    use_pair_common_universe: bool = True

    alpha_horizon_minutes: float = 5.0
    alpha_target_corr: float = 0.10
    alpha_decay_half_life_minutes: float = 5.0
    alpha_horizon_grid_minutes: tuple[float, ...] = (1.0, 5.0, 10.0)
    alpha_rho_grid: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50)
    alpha_decay_grid_minutes: tuple[float, ...] = (1.0, 5.0, 30.0, 60.0)

    run_ow_strategy: bool = True
    run_reduced_form_evaluator: bool = True
    run_stress_tests: bool = True
    signal_delay_minutes: float = 1.0
    forced_liquidation_time: str = "12:00:00"
    liquidation_mode: str = "both"
    resume_alpha_after_liquidation: bool = False
    liquidation_priority_over_alpha: bool = True
    force_eod_liquidation_ignore_caps: bool = False
    mark_residual_inventory_to_market: bool = True
    carry_residual_overnight: bool = True
    max_liquidation_participation_rate: float | None = None
    orderflow_scenario_convention: str = "market_plus_strategy"

    max_pairs: int | None = None
    max_rows_per_pair: int | None = None
    save_pair_level_trades: bool = True
    allow_missing_params: bool = False
    use_x_flow_lambda_proxy: bool = False
    skip_stress: bool = False
    skip_sensitivity: bool = False
    max_participation_rate_per_trade: float | None = 0.01
    max_abs_position_adv_fraction: float | None = 0.05
    max_abs_trade_adv_fraction: float | None = 0.01
    target_impact_scale: float = 1.0
    alpha_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate integration settings."""

        if self.run_mode not in {"single_pair", "all_pairs"}:
            raise ValueError("run_mode must be 'single_pair' or 'all_pairs'")
        if self.orderflow_scenario_convention != "market_plus_strategy":
            raise ValueError("only orderflow_scenario_convention='market_plus_strategy' is implemented")
        if self.liquidation_mode not in {"hard_block", "capped_with_residual", "both"}:
            raise ValueError("liquidation_mode must be 'hard_block', 'capped_with_residual', or 'both'")
        if self.max_liquidation_participation_rate is not None and self.max_liquidation_participation_rate <= 0:
            raise ValueError("max_liquidation_participation_rate must be None or positive")
        for value in [self.alpha_horizon_minutes, self.alpha_decay_half_life_minutes]:
            if value <= 0:
                raise ValueError("alpha horizons and half-lives must be positive")
        if not 0 < self.alpha_target_corr < 1:
            raise ValueError("alpha_target_corr must be in (0, 1)")
        if any(v <= 0 for v in self.alpha_horizon_grid_minutes + self.alpha_decay_grid_minutes):
            raise ValueError("all grid horizon/half-life values must be positive")
        if any(r <= 0 or r >= 1 for r in self.alpha_rho_grid):
            raise ValueError("all alpha rho grid values must be in (0, 1)")
        if self.max_pairs is not None and self.max_pairs <= 0:
            raise ValueError("max_pairs must be None or positive")
        if self.max_rows_per_pair is not None and self.max_rows_per_pair <= 0:
            raise ValueError("max_rows_per_pair must be None or positive")
        for name in [
            "max_participation_rate_per_trade",
            "max_abs_position_adv_fraction",
            "max_abs_trade_adv_fraction",
        ]:
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be None or positive")
        if self.target_impact_scale <= 0:
            raise ValueError("target_impact_scale must be positive")
        if self.alpha_scale <= 0:
            raise ValueError("alpha_scale must be positive")
