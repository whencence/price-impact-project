"""Section 2.7 sensitivity analysis and stress testing."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.ow_strategy import (
    compute_impact_beta,
    compute_target_impact,
    normalize_signed_volume,
    prepare_strategy_data,
    run_ow_strategy,
)
from src.scaling_factors import compute_or_load_scaling_factors
from src.strategy_config import OWStrategyConfig
from src.stress_config import SensitivityConfig, StressRunConfig, StressTestConfig
from src.stress_metrics import (
    compare_to_baseline,
    extract_strategy_summary,
    save_cumulative_wealth_plot,
    save_sensitivity_summary,
    save_stress_summary,
    save_summary_bar_plots,
)


def _warn(message: str) -> None:
    print(f"WARNING: {message}")


def _scenario_alpha_col(half_life: float) -> str:
    return f"alpha_state_H{int(half_life) if float(half_life).is_integer() else half_life}m"


def load_alpha_input(path: Path) -> pd.DataFrame:
    """Load and validate the strategy alpha input."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"alpha input not found: {path}")
    df = pd.read_csv(path)
    required = {"date", "time", "timestamp", "stock", "mid", "alpha_for_strategy"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"alpha input missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("alpha input has unparseable timestamps")
    return df


def make_ow_config_for_scenario(
    base_ow_config: OWStrategyConfig,
    impact_lambda_multiplier: float = 1.0,
    impact_half_life_minutes: float | None = None,
    alpha_col: str = "alpha_for_strategy",
) -> OWStrategyConfig:
    """Create an OW config for a scenario."""

    return replace(
        base_ow_config,
        impact_lambda=base_ow_config.impact_lambda * impact_lambda_multiplier,
        impact_half_life_minutes=impact_half_life_minutes or base_ow_config.impact_half_life_minutes,
        alpha_col=alpha_col,
    )


def run_strategy_for_scenario(
    alpha_df: pd.DataFrame,
    model_name: str,
    scenario_name: str,
    scenario_type: str,
    ow_config: OWStrategyConfig | None = None,
    reduced_form_config: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a strategy scenario and return trades plus summary."""

    metadata = metadata or {}
    if model_name == "OW":
        if ow_config is None:
            raise ValueError("ow_config is required for OW scenarios")
        trades = run_ow_strategy(alpha_df, ow_config)
    elif model_name == "reduced_form":
        try:
            from src.reduced_form_strategy import run_reduced_form_strategy
        except Exception as exc:  # pragma: no cover - defensive integration hook
            raise NotImplementedError("reduced_form module is not available") from exc
        if reduced_form_config is None:
            raise ValueError("reduced_form_config is required for reduced_form scenarios")
        trades = run_reduced_form_strategy(alpha_df, reduced_form_config)
    else:
        raise ValueError(f"unsupported model_name: {model_name}")
    return trades, extract_strategy_summary(trades, scenario_name, scenario_type, metadata)


def run_impact_parameter_sensitivity(
    alpha_df: pd.DataFrame,
    base_ow_config: OWStrategyConfig,
    sensitivity_config: SensitivityConfig,
    output_dir: Path,
) -> pd.DataFrame:
    """Run OW sensitivity over lambda multiplier and impact half-life."""

    output_dir = Path(output_dir)
    rows = []
    count = 0
    for mult in sensitivity_config.impact_lambda_multiplier_grid:
        if sensitivity_config.max_scenarios and count >= sensitivity_config.max_scenarios:
            break
        name = f"sensitivity_lambda_mult_{mult:g}_OW"
        cfg = make_ow_config_for_scenario(
            base_ow_config,
            impact_lambda_multiplier=mult,
            impact_half_life_minutes=sensitivity_config.baseline_impact_half_life_minutes,
        )
        trades, summary = run_strategy_for_scenario(
            alpha_df,
            "OW",
            name,
            "impact_parameter_sensitivity",
            cfg,
            metadata={"impact_lambda_multiplier": mult, "impact_half_life_minutes": cfg.impact_half_life_minutes},
        )
        if sensitivity_config.save_scenario_trades:
            trades.to_csv(output_dir / f"{name}_trades.csv", index=False)
        rows.append(summary)
        count += 1
    for half_life in sensitivity_config.impact_half_life_grid_minutes:
        if sensitivity_config.max_scenarios and count >= sensitivity_config.max_scenarios:
            break
        name = f"sensitivity_impact_H{half_life:g}m_OW"
        cfg = make_ow_config_for_scenario(
            base_ow_config,
            impact_lambda_multiplier=sensitivity_config.baseline_impact_lambda_multiplier,
            impact_half_life_minutes=half_life,
        )
        trades, summary = run_strategy_for_scenario(
            alpha_df,
            "OW",
            name,
            "impact_parameter_sensitivity",
            cfg,
            metadata={"impact_lambda_multiplier": 1.0, "impact_half_life_minutes": half_life},
        )
        if sensitivity_config.save_scenario_trades:
            trades.to_csv(output_dir / f"{name}_trades.csv", index=False)
        rows.append(summary)
        count += 1
    return pd.DataFrame(rows)


def run_alpha_decay_sensitivity(
    baseline_alpha_df: pd.DataFrame,
    base_ow_config: OWStrategyConfig,
    sensitivity_config: SensitivityConfig,
    output_dir: Path,
) -> pd.DataFrame:
    """Run OW strategy for available alpha decay state columns."""

    rows = []
    for half_life in sensitivity_config.alpha_decay_half_life_grid_minutes:
        col = _scenario_alpha_col(half_life)
        if col not in baseline_alpha_df.columns:
            _warn(f"alpha decay column missing, skipping: {col}")
            continue
        name = f"sensitivity_alpha_decay_H{half_life:g}m_OW"
        cfg = make_ow_config_for_scenario(base_ow_config, alpha_col=col)
        trades, summary = run_strategy_for_scenario(
            baseline_alpha_df,
            "OW",
            name,
            "alpha_decay_sensitivity",
            cfg,
            metadata={"alpha_decay_half_life_minutes": half_life},
        )
        if sensitivity_config.save_scenario_trades:
            trades.to_csv(Path(output_dir) / f"{name}_trades.csv", index=False)
        rows.append(summary)
    return pd.DataFrame(rows)


def run_alpha_strength_and_horizon_sensitivity(
    base_ow_config: OWStrategyConfig,
    sensitivity_config: SensitivityConfig,
    output_dir: Path,
    scenario_dir: Path | None = None,
) -> pd.DataFrame:
    """Run h/rho alpha sensitivity from existing scenario alpha files if present."""

    output_dir = Path(output_dir)
    scenario_dir = scenario_dir or output_dir.parent / "alphas" / "scenarios"
    rows = []
    skipped = []
    for h in sensitivity_config.horizon_minutes_grid:
        for rho in sensitivity_config.rho_grid:
            rho_key = f"{int(round(rho * 100)):03d}"
            h_key = f"{h:g}".replace(".", "p")
            candidates = list(Path(scenario_dir).glob(f"*h{h_key}m*rho{rho_key}*H5m*.csv"))
            if not candidates:
                skipped.append({"scenario_name": f"sensitivity_rho_{rho:g}_h{h:g}_Halpha5_OW", "reason": "scenario alpha file not found"})
                continue
            alpha_df = load_alpha_input(candidates[0])
            name = f"sensitivity_rho_{rho:g}_h{h:g}_Halpha5_OW"
            trades, summary = run_strategy_for_scenario(
                alpha_df,
                "OW",
                name,
                "alpha_horizon_rho_sensitivity",
                base_ow_config,
                metadata={"horizon_minutes": h, "target_corr": rho},
            )
            if sensitivity_config.save_scenario_trades:
                trades.to_csv(output_dir / f"{name}_trades.csv", index=False)
            rows.append(summary)
    if skipped:
        pd.DataFrame(skipped).to_csv(output_dir / "skipped_scenarios.csv", index=False)
        _warn("Alpha h/rho sensitivity requires scenario alpha inputs. Missing scenarios were recorded.")
    return pd.DataFrame(rows)


def apply_clock_time_signal_delay(
    df: pd.DataFrame,
    alpha_col: str = "alpha_for_strategy",
    delay_minutes: float = 1.0,
    stock_col: str = "stock",
    date_col: str = "date",
    timestamp_col: str = "timestamp",
    output_col: str | None = None,
    tolerance_seconds: float | None = None,
) -> pd.DataFrame:
    """Delay alpha by clock time within each stock/date group."""

    if delay_minutes < 0:
        raise ValueError("delay_minutes must be non-negative")
    if alpha_col not in df.columns:
        raise ValueError(f"alpha column not found: {alpha_col}")
    output_col = output_col or f"alpha_delayed_{delay_minutes:g}m"
    pieces = []
    for _, group in df.copy().groupby([stock_col, date_col], sort=False):
        g = group.sort_values(timestamp_col).copy()
        ts = pd.to_datetime(g[timestamp_col]).to_numpy(dtype="datetime64[ns]")
        target = pd.to_datetime(g[timestamp_col]) - pd.Timedelta(minutes=delay_minutes)
        idx = np.searchsorted(ts, target.to_numpy(dtype="datetime64[ns]"), side="right") - 1
        valid = idx >= 0
        alpha_values = g[alpha_col].to_numpy()
        delayed = np.zeros(len(g), dtype=float)
        delayed_ts = np.full(len(g), np.datetime64("NaT"), dtype="datetime64[ns]")
        delayed[valid] = alpha_values[idx[valid]]
        delayed_ts[valid] = ts[idx[valid]]
        age = (pd.to_datetime(g[timestamp_col]).to_numpy(dtype="datetime64[ns]") - delayed_ts) / np.timedelta64(1, "s")
        if tolerance_seconds is not None:
            too_old = valid & (age > tolerance_seconds)
            delayed[too_old] = 0.0
            delayed_ts[too_old] = np.datetime64("NaT")
            age[too_old] = np.nan
        g[output_col] = delayed
        g["alpha_delay_minutes"] = delay_minutes
        g["delayed_alpha_timestamp"] = delayed_ts
        g["delayed_alpha_age_seconds"] = age
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True).sort_values([stock_col, date_col, timestamp_col]).reset_index(drop=True)


def run_signal_delay_stress(
    alpha_df: pd.DataFrame,
    base_ow_config: OWStrategyConfig,
    stress_config: StressTestConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run one-minute clock-time signal delay stress."""

    delayed_col = f"alpha_delayed_{stress_config.signal_delay_minutes:g}m"
    delayed = apply_clock_time_signal_delay(alpha_df, base_ow_config.alpha_col, stress_config.signal_delay_minutes, output_col=delayed_col)
    cfg = replace(base_ow_config, alpha_col=delayed_col)
    trades, summary = run_strategy_for_scenario(
        delayed,
        "OW",
        f"delay_{stress_config.signal_delay_minutes:g}m_OW",
        "signal_delay_stress",
        cfg,
        metadata={"signal_delay_minutes": stress_config.signal_delay_minutes},
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trades.to_csv(Path(output_dir) / "signal_delay_trades.csv", index=False)
    pd.DataFrame([summary]).to_csv(Path(output_dir) / "signal_delay_summary.csv", index=False)
    return trades, summary


def run_ow_strategy_with_forced_liquidation(
    alpha_df: pd.DataFrame,
    base_ow_config: OWStrategyConfig,
    liquidation_time: str = "12:00:00",
    resume_after_liquidation: bool = False,
) -> pd.DataFrame:
    """Run OW with a block liquidation at the first timestamp >= liquidation_time."""

    prepared = compute_target_impact(prepare_strategy_data(alpha_df, base_ow_config), base_ow_config)
    beta = compute_impact_beta(base_ow_config)
    threshold = pd.to_datetime(liquidation_time).time()
    rows = []
    for _, group in prepared.groupby([base_ow_config.stock_col, base_ow_config.date_col], sort=False):
        position_prev = float(base_ow_config.initial_position)
        impact_after_prev = float(base_ow_config.initial_impact)
        liquidated = False
        trigger_done = False
        for idx, row in group.iterrows():
            dt = float(row["dt_minutes"])
            decay = float(np.exp(-beta * dt))
            impact_before = decay * impact_after_prev
            sigma = float(row["sigma"]) if pd.notna(row.get("sigma")) else np.nan
            adv = float(row["ADV"]) if pd.notna(row.get("ADV")) else np.nan
            has_scaling = bool(row.get("has_scaling", False))
            is_trigger = (not trigger_done) and pd.to_datetime(row[base_ow_config.timestamp_col]).time() >= threshold
            if liquidated and not resume_after_liquidation:
                signed_volume = 0.0
                position_after = 0.0
                is_forced = False
            elif is_trigger:
                signed_volume = -position_prev
                position_after = 0.0
                is_forced = True
                trigger_done = True
                liquidated = True
            elif has_scaling:
                required = (float(row["target_impact"]) - impact_before) / base_ow_config.impact_lambda
                signed_volume = required * adv / sigma if sigma > 0 and adv > 0 else 0.0
                position_after = position_prev + signed_volume
                is_forced = False
            else:
                signed_volume = 0.0
                position_after = position_prev
                is_forced = False
            normalized = float(normalize_signed_volume(signed_volume, sigma, adv, base_ow_config.impact_model_type)) if has_scaling else 0.0
            if not np.isfinite(normalized):
                normalized = 0.0
            impact_after = impact_before + base_ow_config.impact_lambda * normalized
            gross_pnl = position_prev * float(row["delta_mid"])
            quad = 0.5 * base_ow_config.impact_lambda * normalized**2
            signed_cost = impact_before * normalized + quad
            rec = row.to_dict()
            rec.update(
                {
                    "decay_factor": decay,
                    "impact_before_trade": impact_before,
                    "normalized_trade": normalized,
                    "signed_volume": signed_volume,
                    "trade": signed_volume,
                    "position_before": position_prev,
                    "position_after": position_after,
                    "impact_after_trade": impact_after,
                    "participation_rate": abs(signed_volume) / adv if np.isfinite(adv) and adv > 0 else np.nan,
                    "gross_pnl": gross_pnl,
                    "quadratic_impact_cost_normalized": quad,
                    "signed_impact_cost_normalized": signed_cost,
                    "quadratic_impact_cost": quad,
                    "signed_impact_cost": signed_cost,
                    "net_pnl": gross_pnl - signed_cost,
                    "is_forced_liquidation": is_forced,
                    "forced_liquidation_trade": signed_volume if is_forced else 0.0,
                    "is_liquidation": False,
                }
            )
            rows.append(rec)
            position_prev = position_after
            impact_after_prev = impact_after
    out = pd.DataFrame(rows).sort_values([base_ow_config.stock_col, base_ow_config.date_col, base_ow_config.timestamp_col]).reset_index(drop=True)
    out["cumulative_wealth"] = out["net_pnl"].cumsum()
    out["cumulative_gross_pnl"] = out["gross_pnl"].cumsum()
    return out


def run_forced_liquidation_stress(
    alpha_df: pd.DataFrame,
    base_ow_config: OWStrategyConfig,
    stress_config: StressTestConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run forced liquidation stress."""

    trades = run_ow_strategy_with_forced_liquidation(
        alpha_df,
        base_ow_config,
        stress_config.forced_liquidation_time,
        stress_config.resume_after_liquidation,
    )
    events = trades.loc[trades["is_forced_liquidation"].astype(bool)]
    metadata = {
        "forced_liquidation_time": stress_config.forced_liquidation_time,
        "resume_after_liquidation": stress_config.resume_after_liquidation,
        "number_of_liquidation_events": int(len(events)),
        "mean_abs_liquidation_trade": float(events["forced_liquidation_trade"].abs().mean()) if len(events) else 0.0,
        "max_abs_liquidation_trade": float(events["forced_liquidation_trade"].abs().max()) if len(events) else 0.0,
        "liquidation_cost_total": float(events.get("signed_impact_cost_normalized", pd.Series(dtype=float)).sum()) if len(events) else 0.0,
    }
    summary = extract_strategy_summary(trades, "forced_liq_1200_stop_OW", "forced_liquidation_stress", metadata)
    trades.to_csv(Path(output_dir) / "forced_liquidation_trades.csv", index=False)
    pd.DataFrame([summary]).to_csv(Path(output_dir) / "forced_liquidation_summary.csv", index=False)
    return trades, summary


def recompute_ow_wealth_given_trades(
    trades_df: pd.DataFrame,
    true_ow_config: OWStrategyConfig,
) -> pd.DataFrame:
    """Keep signed-volume path fixed and recompute OW wealth under true parameters."""

    out_rows = []
    beta = compute_impact_beta(true_ow_config)
    df = trades_df.copy().sort_values([true_ow_config.stock_col, true_ow_config.date_col, true_ow_config.timestamp_col])
    for _, group in df.groupby([true_ow_config.stock_col, true_ow_config.date_col], sort=False):
        impact_after_prev = float(true_ow_config.initial_impact)
        position_prev = float(true_ow_config.initial_position)
        for _, row in group.iterrows():
            dt = float(row["dt_minutes"])
            decay = float(np.exp(-beta * dt))
            impact_before = decay * impact_after_prev
            signed_volume = float(row["signed_volume"] if "signed_volume" in row else row["trade"])
            sigma = float(row["sigma"]) if pd.notna(row.get("sigma")) else np.nan
            adv = float(row["ADV"]) if pd.notna(row.get("ADV")) else np.nan
            normalized = float(normalize_signed_volume(signed_volume, sigma, adv, true_ow_config.impact_model_type)) if sigma > 0 and adv > 0 else 0.0
            if not np.isfinite(normalized):
                normalized = 0.0
            impact_after = impact_before + true_ow_config.impact_lambda * normalized
            gross = position_prev * float(row["delta_mid"])
            quad = 0.5 * true_ow_config.impact_lambda * normalized**2
            signed_cost = impact_before * normalized + quad
            rec = row.to_dict()
            rec.update(
                {
                    "impact_before_trade_true": impact_before,
                    "normalized_trade_true": normalized,
                    "impact_after_trade_true": impact_after,
                    "gross_pnl_true": gross,
                    "quadratic_impact_cost_true": quad,
                    "signed_impact_cost_true": signed_cost,
                    "net_pnl_true": gross - signed_cost,
                }
            )
            out_rows.append(rec)
            position_prev = float(row["position_after"])
            impact_after_prev = impact_after
    out = pd.DataFrame(out_rows)
    out["cumulative_wealth_true"] = out["net_pnl_true"].cumsum()
    return out


def run_wrong_impact_parameter_stress(
    alpha_df: pd.DataFrame,
    assumed_ow_config: OWStrategyConfig,
    true_ow_config: OWStrategyConfig,
    stress_config: StressTestConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run trades under assumed OW parameters and realize wealth under true parameters."""

    assumed = run_ow_strategy(alpha_df, assumed_ow_config)
    out = recompute_ow_wealth_given_trades(assumed, true_ow_config)
    summary_df = out.copy()
    summary_df["net_pnl"] = summary_df["net_pnl_true"]
    summary_df["gross_pnl"] = summary_df["gross_pnl_true"]
    summary_df["signed_impact_cost"] = summary_df["signed_impact_cost_true"]
    summary_df["quadratic_impact_cost"] = summary_df["quadratic_impact_cost_true"]
    summary_df["impact_after_trade"] = summary_df["impact_after_trade_true"]
    summary_df["cumulative_wealth"] = summary_df["cumulative_wealth_true"]
    metadata = {
        "assumed_lambda": assumed_ow_config.impact_lambda,
        "true_lambda": true_ow_config.impact_lambda,
        "assumed_impact_half_life_minutes": assumed_ow_config.impact_half_life_minutes,
        "true_impact_half_life_minutes": true_ow_config.impact_half_life_minutes,
        "assumed_total_net_pnl": float(assumed["net_pnl"].sum()),
        "true_total_net_pnl": float(out["net_pnl_true"].sum()),
        "model_error_cost": float(out["net_pnl_true"].sum() - assumed["net_pnl"].sum()),
    }
    summary = extract_strategy_summary(summary_df, "wrong_impact_lambda2_H30_OW", "wrong_impact_parameter_stress", metadata)
    out.to_csv(Path(output_dir) / "wrong_impact_trades.csv", index=False)
    pd.DataFrame([summary]).to_csv(Path(output_dir) / "wrong_impact_summary.csv", index=False)
    return out, summary


def run_wrong_impact_model_stress(*_: Any, **__: Any) -> None:
    """Placeholder for future fitted-model wrong-impact stress."""

    _warn("True fitted-model wrong-impact stress skipped because fitted model parameters are not available.")
    return None


def run_all_sensitivity_analyses(
    alpha_df: pd.DataFrame,
    base_ow_config: OWStrategyConfig,
    sensitivity_config: SensitivityConfig,
    output_dir: Path,
) -> pd.DataFrame:
    """Run all available sensitivity analyses."""

    output_dir = Path(output_dir)
    baseline_trades, baseline_summary = run_strategy_for_scenario(
        alpha_df, "OW", "baseline_OW", "baseline", base_ow_config
    )
    baseline_trades.to_csv(output_dir / "baseline_OW_trades.csv", index=False)
    parts = [pd.DataFrame([baseline_summary])]
    parts.append(run_impact_parameter_sensitivity(alpha_df, base_ow_config, sensitivity_config, output_dir))
    parts.append(run_alpha_decay_sensitivity(alpha_df, base_ow_config, sensitivity_config, output_dir))
    alpha_hrho = run_alpha_strength_and_horizon_sensitivity(base_ow_config, sensitivity_config, output_dir)
    if not alpha_hrho.empty:
        alpha_hrho.to_csv(output_dir / "alpha_h_rho_sensitivity_summary.csv", index=False)
        parts.append(alpha_hrho)
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    out = compare_to_baseline(out, "baseline_OW")
    save_sensitivity_summary(out, output_dir)
    return out


def run_all_stress_tests(
    alpha_df: pd.DataFrame,
    base_ow_config: OWStrategyConfig,
    sensitivity_config: SensitivityConfig,
    stress_config: StressTestConfig,
    run_config: StressRunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run baseline, sensitivity analyses, and requested stress tests."""

    output_dir = Path(run_config.output_dir)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    sensitivity = run_all_sensitivity_analyses(alpha_df, base_ow_config, sensitivity_config, output_dir)
    baseline_trades = run_ow_strategy(alpha_df, base_ow_config)
    stress_rows = [extract_strategy_summary(baseline_trades, "baseline_OW", "baseline")]
    wealth_plots = {"baseline": baseline_trades}

    if stress_config.run_signal_delay:
        delayed_trades, delayed_summary = run_signal_delay_stress(alpha_df, base_ow_config, stress_config, output_dir)
        stress_rows.append(delayed_summary)
        wealth_plots["delayed"] = delayed_trades
        save_cumulative_wealth_plot({"baseline": baseline_trades, "delayed": delayed_trades}, fig_dir / "signal_delay_wealth.png", "Signal delay stress")
    if stress_config.run_forced_liquidation:
        forced_trades, forced_summary = run_forced_liquidation_stress(alpha_df, base_ow_config, stress_config, output_dir)
        stress_rows.append(forced_summary)
        wealth_plots["forced liquidation"] = forced_trades
        save_cumulative_wealth_plot({"baseline": baseline_trades, "forced liquidation": forced_trades}, fig_dir / "forced_liquidation_wealth.png", "Forced liquidation stress")
    if stress_config.run_wrong_impact_params:
        assumed_cfg = make_ow_config_for_scenario(
            base_ow_config,
            stress_config.wrong_impact_lambda_multiplier_assumed,
            stress_config.wrong_impact_half_life_assumed_minutes,
        )
        true_cfg = make_ow_config_for_scenario(
            base_ow_config,
            stress_config.wrong_impact_lambda_multiplier_true,
            stress_config.wrong_impact_half_life_true_minutes,
        )
        wrong_trades, wrong_summary = run_wrong_impact_parameter_stress(alpha_df, assumed_cfg, true_cfg, stress_config, output_dir)
        stress_rows.append(wrong_summary)
        if "cumulative_wealth_true" in wrong_trades.columns:
            plot_df = wrong_trades.rename(columns={"cumulative_wealth_true": "cumulative_wealth"})
            wealth_plots["wrong impact true"] = plot_df
        save_cumulative_wealth_plot(
            {"assumed": wrong_trades, "true": wrong_trades.rename(columns={"cumulative_wealth_true": "cumulative_wealth"})},
            fig_dir / "wrong_impact_wealth.png",
            "Wrong impact parameter stress",
        )
    if stress_config.run_wrong_impact_model:
        run_wrong_impact_model_stress()

    stress = compare_to_baseline(pd.DataFrame(stress_rows), "baseline_OW")
    save_stress_summary(stress, output_dir)
    save_cumulative_wealth_plot(wealth_plots, fig_dir / "baseline_vs_stress_cumulative_wealth.png", "Baseline vs stress cumulative wealth")
    all_summary = pd.concat([sensitivity, stress], ignore_index=True).drop_duplicates("scenario_name")
    all_summary.to_csv(output_dir / "all_scenarios_summary.csv", index=False)
    save_summary_bar_plots(all_summary, fig_dir)
    _save_sensitivity_plots(sensitivity, fig_dir)
    return sensitivity, stress


def _save_sensitivity_plots(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Save robust sensitivity plots when matching data exists."""

    fig_dir = Path(fig_dir)
    impact = summary[summary["scenario_type"].eq("impact_parameter_sensitivity")].copy()
    if not impact.empty and "impact_lambda_multiplier" in impact.columns:
        by_lambda = impact.dropna(subset=["impact_lambda_multiplier"])
        if not by_lambda.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(by_lambda["impact_lambda_multiplier"], by_lambda["total_net_pnl"], marker="o", label="net pnl")
            ax.set_title("Impact lambda sensitivity")
            ax.set_xlabel("lambda multiplier")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / "impact_lambda_sensitivity.png", dpi=150)
            plt.close(fig)
    if not impact.empty and "impact_half_life_minutes" in impact.columns:
        by_h = impact.dropna(subset=["impact_half_life_minutes"])
        if not by_h.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(by_h["impact_half_life_minutes"], by_h["total_net_pnl"], marker="o")
            ax.set_title("Impact half-life sensitivity")
            ax.set_xlabel("H_I minutes")
            fig.tight_layout()
            fig.savefig(fig_dir / "impact_half_life_sensitivity.png", dpi=150)
            plt.close(fig)
    decay = summary[summary["scenario_type"].eq("alpha_decay_sensitivity")]
    if not decay.empty and "alpha_decay_half_life_minutes" in decay.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(decay["alpha_decay_half_life_minutes"], decay["total_net_pnl"], marker="o", label="net pnl")
        ax.plot(decay["alpha_decay_half_life_minutes"], decay["total_turnover"], marker="o", label="turnover")
        ax.set_title("Alpha decay sensitivity")
        ax.set_xlabel("H_alpha minutes")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "alpha_decay_sensitivity.png", dpi=150)
        plt.close(fig)
