"""Deep implementation audit for integrated OW and fitted-regression strategies.

The audit is intentionally diagnostic rather than another backtest. It checks
whether the OW target-impact strategy, teammate fitted-regression impact
evaluator, and fitted-regression proxy strategy are internally consistent and
whether large cost differences are explained by sizing, turnover, or feature
extrapolation.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fitted_regression_features import (
    compute_training_scales,
    prepare_model_frame,
)
from src.fitted_regression_params import load_reduced_form_params
from src.integration_config import TeammateDataConfig
from src.reduced_form_regression_evaluator import (
    add_strategy_trade_to_orderflow,
    evaluate_marginal_impact_from_strategy_trades,
    prepare_scenario_model_frame,
)
from src.teammate_integration import (
    load_monthly_bin_data_for_pair,
    load_rolling_pair_summary,
    parse_pair_universe,
    resolve_project_root,
)


STATUS_ORDER = {"FAIL": 0, "WARN": 1, "SKIP": 2, "PASS": 3}


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def _safe_sum(df: pd.DataFrame | None, col: str) -> float:
    if df is None or col not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").sum())


def _safe_mean(df: pd.DataFrame | None, col: str) -> float:
    if df is None or col not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def _finite_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return np.nan
    return float(numerator / denominator)


def _status_row(check: str, status: str, message: str, **extra: Any) -> dict[str, Any]:
    row = {"check": check, "status": status, "message": message}
    row.update(extra)
    return row


def _fmt(value: Any, precision: int = 4) -> str:
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(value):
        return "n/a"
    if abs(value) >= 1e5 or (abs(value) < 1e-3 and value != 0):
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def _ensure_timestamp(df: pd.DataFrame, timestamp_col: str = "datetime") -> pd.DataFrame:
    out = df.copy()
    if timestamp_col not in out.columns and "timestamp" in out.columns:
        out[timestamp_col] = out["timestamp"]
    if timestamp_col in out.columns:
        out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    return out


def _notional_turnover(df: pd.DataFrame | None) -> float:
    if df is None or len(df) == 0:
        return np.nan
    if "notional_turnover" in df.columns:
        return float(pd.to_numeric(df["notional_turnover"], errors="coerce").sum())
    if {"signed_volume", "mid"}.issubset(df.columns):
        return float((pd.to_numeric(df["signed_volume"], errors="coerce").abs() * pd.to_numeric(df["mid"], errors="coerce")).sum())
    return np.nan


def _total_abs_volume(df: pd.DataFrame | None) -> float:
    if df is None or "signed_volume" not in df.columns:
        return np.nan
    return float(pd.to_numeric(df["signed_volume"], errors="coerce").abs().sum())


def _quantiles(series: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return {"min": np.nan, "p1": np.nan, "median": np.nan, "mean": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan}
    return {
        "min": float(x.min()),
        "p1": float(x.quantile(0.01)),
        "median": float(x.median()),
        "mean": float(x.mean()),
        "p95": float(x.quantile(0.95)),
        "p99": float(x.quantile(0.99)),
        "max": float(x.max()),
    }


def _portfolio_cumsum(df: pd.DataFrame | None, value_col: str) -> pd.DataFrame:
    if df is None or value_col not in df.columns:
        return pd.DataFrame(columns=["datetime", value_col, f"cum_{value_col}"])
    out = _ensure_timestamp(df, "datetime")
    if "datetime" not in out.columns:
        return pd.DataFrame(columns=["datetime", value_col, f"cum_{value_col}"])
    ts = (
        out.dropna(subset=["datetime"])
        .groupby("datetime", as_index=False)[value_col]
        .sum()
        .sort_values("datetime")
    )
    ts[f"cum_{value_col}"] = pd.to_numeric(ts[value_col], errors="coerce").fillna(0.0).cumsum()
    return ts


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _hist_compare(
    series_a: pd.Series,
    series_b: pd.Series,
    label_a: str,
    label_b: str,
    title: str,
    xlabel: str,
    path: Path,
    bins: int = 80,
    log_x: bool = False,
) -> None:
    a = pd.to_numeric(series_a, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.to_numeric(series_b, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if log_x:
        a = a[a > 0]
        b = b[b > 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(a):
        ax.hist(a, bins=bins, alpha=0.55, label=label_a)
    if len(b):
        ax.hist(b, bins=bins, alpha=0.55, label=label_b)
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Rows")
    ax.legend()
    _save_fig(fig, path)


def _load_context(experiment_path: Path, pair_id: int) -> dict[str, Any]:
    root = resolve_project_root()
    exp = (root / experiment_path).resolve() if not experiment_path.is_absolute() else experiment_path.resolve()
    pair_dir = exp / f"pair_{pair_id}"
    metadata_path = exp / "metadata" / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    ow_trades = _read_csv(pair_dir / "my_ow_trades.csv")
    if ow_trades is None:
        ow_trades = _read_csv(pair_dir / "trades" / "baseline_ow_trades.csv")
    proxy_trades = _read_csv(pair_dir / "fitted_proxy_strategy_trades.csv")
    if proxy_trades is None:
        proxy_trades = _read_csv(pair_dir / "trades" / "fitted_proxy_strategy_trades.csv")
    strategy_summary = _read_csv(exp / "all_pairs_strategy_summary.csv")
    if strategy_summary is None:
        strategy_summary = _read_csv(exp / "tables" / "strategy_summary.csv")
    proxy_summary = _read_csv(exp / "all_pairs_fitted_proxy_summary.csv")
    if proxy_summary is None:
        proxy_summary = _read_csv(exp / "tables" / "fitted_proxy_summary.csv")
    context = {
        "root": root,
        "experiment_path": exp,
        "pair_dir": pair_dir,
        "metadata": metadata,
        "ow_trades": ow_trades,
        "ow_eval": _read_csv(pair_dir / "ow_transient_regression_evaluator.csv"),
        "rf_eval": _read_csv(pair_dir / "reduced_form_regression_evaluator.csv"),
        "proxy_trades": proxy_trades,
        "strategy_summary": strategy_summary,
        "proxy_summary": proxy_summary,
    }
    return context


def _load_pair_raw_data(context: dict[str, Any], pair_id: int) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = context["root"]
    metadata = context["metadata"]
    data_config = TeammateDataConfig()
    pairs = load_rolling_pair_summary(data_config, root)
    pair_row = pairs[pairs["pair_id"].eq(pair_id)]
    if pair_row.empty:
        raise ValueError(f"pair_id={pair_id} not found in rolling pair summary")
    pair_row = pair_row.iloc[0]
    stocks = parse_pair_universe(pair_row)
    max_rows = metadata.get("max_rows_per_pair")
    train_raw, test_raw = load_monthly_bin_data_for_pair(pair_row, stocks, data_config, sample_nrows=max_rows)
    params_path = root / data_config.reduced_form_params_path
    rf_params = load_reduced_form_params(params_path)
    return pair_row, train_raw, test_raw, rf_params


def trade_size_comparison(ow_df: pd.DataFrame | None, proxy_df: pd.DataFrame | None, out_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []

    def metrics(df: pd.DataFrame | None, name: str) -> dict[str, Any]:
        if df is None or "signed_volume" not in df.columns:
            checks.append(_status_row(f"{name}_trade_size_available", "SKIP", "row-level strategy trades unavailable"))
            return {"strategy_name": name}
        q = pd.to_numeric(df["signed_volume"], errors="coerce")
        abs_q = q.abs()
        participation = (
            pd.to_numeric(df["participation_rate"], errors="coerce")
            if "participation_rate" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        position = (
            pd.to_numeric(df["position_after"], errors="coerce").abs()
            if "position_after" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        adv = (
            pd.to_numeric(df["ADV"], errors="coerce")
            if "ADV" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        pos_over_adv = position / adv.replace(0, np.nan)
        checks.append(_status_row(f"{name}_trade_size_available", "PASS", f"rows={len(df)}"))
        return {
            "strategy_name": name,
            "total_abs_volume": float(abs_q.sum()),
            "total_notional_turnover": _notional_turnover(df),
            "mean_abs_trade": float(abs_q.mean()),
            "median_abs_trade": float(abs_q.median()),
            "p95_abs_trade": float(abs_q.quantile(0.95)),
            "p99_abs_trade": float(abs_q.quantile(0.99)),
            "max_abs_trade": float(abs_q.max()),
            "number_of_trades": int((abs_q > 0).sum()),
            "mean_participation_rate": float(participation.mean(skipna=True)),
            "p95_participation_rate": float(participation.quantile(0.95)),
            "max_participation_rate": float(participation.max(skipna=True)),
            "max_position_over_ADV": float(pos_over_adv.max(skipna=True)),
            "share_trade_clipped": float(pd.Series(df.get("trade_clipped", False)).astype(bool).mean()),
            "share_position_clipped": float(pd.Series(df.get("position_clipped", False)).astype(bool).mean()),
        }

    rows = [metrics(ow_df, "OW_target_impact"), metrics(proxy_df, "fitted_proxy")]
    table = pd.DataFrame(rows)
    if len(table) == 2 and table["total_abs_volume"].notna().all():
        ow = table.iloc[0]
        proxy = table.iloc[1]
        ratios = {
            "strategy_name": "OW_to_proxy_ratio",
            "total_abs_volume": _finite_ratio(ow["total_abs_volume"], proxy["total_abs_volume"]),
            "total_notional_turnover": _finite_ratio(ow["total_notional_turnover"], proxy["total_notional_turnover"]),
            "mean_abs_trade": _finite_ratio(ow["mean_abs_trade"], proxy["mean_abs_trade"]),
        }
        table = pd.concat([table, pd.DataFrame([ratios])], ignore_index=True)
        status = "WARN" if ratios["total_abs_volume"] > 5 else "PASS"
        checks.append(_status_row("ow_vs_proxy_turnover_ratio", status, f"OW/proxy total_abs_volume={_fmt(ratios['total_abs_volume'])}"))
    table.to_csv(out_dir / "audit_trade_size_comparison.csv", index=False)
    return table, checks


def cost_per_turnover_table(
    ow_eval: pd.DataFrame | None,
    rf_eval: pd.DataFrame | None,
    proxy_df: pd.DataFrame | None,
    proxy_rf_eval: pd.DataFrame | None,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []

    def add_eval(name: str, evaluator: str, df: pd.DataFrame | None, cost_col: str, abs_col: str | None, net_col: str, gross_col: str) -> None:
        if df is None:
            rows.append({"strategy_name": name, "evaluator_name": evaluator, "status": "SKIP"})
            return
        total_cost = _safe_sum(df, cost_col)
        total_abs_cost = _safe_sum(df, abs_col) if abs_col else abs(total_cost)
        turnover = _notional_turnover(df)
        gross = _safe_sum(df, gross_col)
        net = _safe_sum(df, net_col)
        rows.append(
            {
                "strategy_name": name,
                "evaluator_name": evaluator,
                "status": "PASS",
                "total_cost": total_cost,
                "total_abs_cost": total_abs_cost,
                "total_notional_turnover": turnover,
                "cost_bps_of_turnover": 10000.0 * total_cost / turnover if np.isfinite(turnover) and turnover > 0 else np.nan,
                "net_pnl": net,
                "gross_pnl": gross,
                "cost_to_gross_pnl_ratio": _finite_ratio(total_cost, gross),
            }
        )

    add_eval("OW_target_impact", "OW_transient_regression", ow_eval, "fitted_impact_cost_signed", "fitted_impact_cost_abs", "net_pnl_fitted_model", "gross_pnl")
    add_eval("OW_target_impact", "reduced_form_regression", rf_eval, "fitted_impact_cost_signed", "fitted_impact_cost_abs", "net_pnl_fitted_model", "gross_pnl")
    add_eval("fitted_proxy", "proxy_internal_quadratic", proxy_df, "fitted_impact_cost", None, "net_pnl", "gross_pnl")
    add_eval("fitted_proxy", "full_reduced_form_regression", proxy_rf_eval, "fitted_impact_cost_signed", "fitted_impact_cost_abs", "net_pnl_fitted_model", "gross_pnl")
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "audit_cost_per_turnover.csv", index=False)
    return table


def proxy_cost_consistency(proxy_df: pd.DataFrame | None, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    if proxy_df is None:
        recon = pd.DataFrame([{"status": "SKIP", "reason": "proxy trades unavailable"}])
        slopes = pd.DataFrame([{"status": "SKIP", "reason": "proxy trades unavailable"}])
        recon.to_csv(out_dir / "audit_proxy_cost_reconstruction.csv", index=False)
        slopes.to_csv(out_dir / "audit_proxy_slope_distribution.csv", index=False)
        checks.append(_status_row("proxy_cost_reconstruction", "SKIP", "proxy trades unavailable"))
        return recon, slopes, checks

    required = {"impact_slope_price_per_share", "signed_volume", "fitted_impact_cost"}
    if not required.issubset(proxy_df.columns):
        missing = sorted(required.difference(proxy_df.columns))
        checks.append(_status_row("proxy_cost_reconstruction", "FAIL", f"missing columns={missing}"))
        return pd.DataFrame(), pd.DataFrame(), checks
    reconstructed = (
        pd.to_numeric(proxy_df["impact_slope_price_per_share"], errors="coerce")
        * pd.to_numeric(proxy_df["signed_volume"], errors="coerce") ** 2
    )
    reported = pd.to_numeric(proxy_df["fitted_impact_cost"], errors="coerce")
    err = (reconstructed - reported).abs()
    recon = pd.DataFrame(
        [
            {
                "status": "PASS" if err.max(skipna=True) < 1e-6 else "FAIL",
                "max_reconstruction_error": float(err.max(skipna=True)),
                "mean_reconstruction_error": float(err.mean(skipna=True)),
                "total_reconstructed_cost": float(reconstructed.sum(skipna=True)),
                "reported_total_cost": float(reported.sum(skipna=True)),
            }
        ]
    )
    status = str(recon.loc[0, "status"])
    checks.append(_status_row("proxy_cost_reconstruction", status, f"max_error={_fmt(recon.loc[0, 'max_reconstruction_error'])}"))

    slope_rows = []
    for col in ["impact_slope_bps_per_share", "impact_slope_price_per_share"]:
        if col in proxy_df.columns:
            row = {"slope_column": col, **_quantiles(proxy_df[col])}
            slope_rows.append(row)
    raw = pd.to_numeric(proxy_df.get("impact_slope_bps_per_share_raw", np.nan), errors="coerce")
    bps = pd.to_numeric(proxy_df.get("impact_slope_bps_per_share", np.nan), errors="coerce")
    floor_share = float((raw <= 0).mean(skipna=True)) if raw.notna().any() else np.nan
    tiny_share = float((bps.abs() < 1e-12).mean(skipna=True)) if bps.notna().any() else np.nan
    slopes = pd.DataFrame(slope_rows)
    slopes["share_slope_floored_or_raw_nonpositive"] = floor_share
    slopes["share_slope_extremely_small"] = tiny_share
    slope_status = "WARN" if np.isfinite(floor_share) and floor_share > 0.05 else "PASS"
    checks.append(_status_row("proxy_slope_floor_share", slope_status, f"share={_fmt(floor_share)}"))
    if len(slopes) and np.isfinite(slopes.loc[0, "median"]) and abs(slopes.loc[0, "median"]) < 1e-12:
        checks.append(_status_row("proxy_median_slope", "WARN", "median impact slope is near zero"))
    else:
        checks.append(_status_row("proxy_median_slope", "PASS", "median impact slope is finite"))
    recon.to_csv(out_dir / "audit_proxy_cost_reconstruction.csv", index=False)
    slopes.to_csv(out_dir / "audit_proxy_slope_distribution.csv", index=False)
    return recon, slopes, checks


def proxy_full_rf_comparison(proxy_df: pd.DataFrame | None, proxy_rf_eval: pd.DataFrame | None, out_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    if proxy_df is None or proxy_rf_eval is None:
        table = pd.DataFrame([{"status": "SKIP", "reason": "proxy trades or full RF evaluation unavailable"}])
        table.to_csv(out_dir / "audit_proxy_internal_vs_full_rf_evaluator.csv", index=False)
        checks.append(_status_row("proxy_internal_vs_full_rf", "SKIP", "proxy trades or full RF evaluation unavailable"))
        return table, checks
    internal_cost = _safe_sum(proxy_df, "fitted_impact_cost")
    full_cost = _safe_sum(proxy_rf_eval, "fitted_impact_cost_signed")
    internal_net = _safe_sum(proxy_df, "net_pnl")
    full_net = _safe_sum(proxy_rf_eval, "net_pnl_fitted_model")
    internal_turnover = _notional_turnover(proxy_df)
    full_turnover = _notional_turnover(proxy_rf_eval)
    ratio = _finite_ratio(full_cost, internal_cost)
    status = "PASS"
    if np.isfinite(ratio) and (ratio > 5.0 or ratio < 0.2):
        status = "WARN"
    table = pd.DataFrame(
        [
            {
                "status": status,
                "total_proxy_internal_cost": internal_cost,
                "total_full_rf_cost_on_proxy": full_cost,
                "ratio_full_rf_to_proxy_internal": ratio,
                "net_pnl_proxy_internal": internal_net,
                "net_pnl_proxy_under_full_rf_evaluator": full_net,
                "mean_abs_marginal_impact_bps_on_proxy_trades": float(pd.to_numeric(proxy_rf_eval["marginal_impact_bps"], errors="coerce").abs().mean()),
                "cost_bps_of_turnover_internal": 10000.0 * internal_cost / internal_turnover if np.isfinite(internal_turnover) and internal_turnover > 0 else np.nan,
                "cost_bps_of_turnover_full_rf": 10000.0 * full_cost / full_turnover if np.isfinite(full_turnover) and full_turnover > 0 else np.nan,
            }
        ]
    )
    table.to_csv(out_dir / "audit_proxy_internal_vs_full_rf_evaluator.csv", index=False)
    checks.append(_status_row("proxy_internal_vs_full_rf", status, f"full/internal cost ratio={_fmt(ratio)}"))
    return table, checks


def feature_extrapolation(
    train_raw: pd.DataFrame | None,
    test_raw: pd.DataFrame | None,
    ow_df: pd.DataFrame | None,
    proxy_df: pd.DataFrame | None,
    out_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    if train_raw is None or test_raw is None:
        table = pd.DataFrame([{"status": "SKIP", "reason": "raw train/test data unavailable"}])
        table.to_csv(out_dir / "audit_feature_extrapolation.csv", index=False)
        return table, [_status_row("feature_extrapolation_available", "SKIP", "raw train/test data unavailable")]
    scales = compute_training_scales(train_raw)
    train_frame = prepare_model_frame(train_raw, scales)
    rows = []
    for label, trades in [("OW_target_impact", ow_df), ("fitted_proxy", proxy_df)]:
        if trades is None:
            continue
        scenario_raw = add_strategy_trade_to_orderflow(test_raw, trades)
        scenario_frame = prepare_scenario_model_frame(scenario_raw, scales, use_scenario_orderflow=True)
        for col in ["x_flow", "x_flow_depth"]:
            train_abs = pd.to_numeric(train_frame[col], errors="coerce").abs().replace([np.inf, -np.inf], np.nan).dropna()
            scen_abs = pd.to_numeric(scenario_frame[col], errors="coerce").abs().replace([np.inf, -np.inf], np.nan).dropna()
            train_p99 = float(train_abs.quantile(0.99)) if len(train_abs) else np.nan
            train_max = float(train_abs.max()) if len(train_abs) else np.nan
            share_gt_p99 = float((scen_abs > train_p99).mean()) if np.isfinite(train_p99) and len(scen_abs) else np.nan
            share_gt_max = float((scen_abs > train_max).mean()) if np.isfinite(train_max) and len(scen_abs) else np.nan
            status = "WARN" if (np.isfinite(share_gt_p99) and share_gt_p99 > 0.05) or (np.isfinite(share_gt_max) and share_gt_max > 0.01) else "PASS"
            rows.append(
                {
                    "strategy_name": label,
                    "feature": col,
                    "train_p99_abs": train_p99,
                    "train_max_abs": train_max,
                    "scenario_p99_abs": float(scen_abs.quantile(0.99)) if len(scen_abs) else np.nan,
                    "scenario_max_abs": float(scen_abs.max()) if len(scen_abs) else np.nan,
                    "share_abs_scenario_gt_train_p99": share_gt_p99,
                    "share_abs_scenario_gt_train_max": share_gt_max,
                    "status": status,
                }
            )
            checks.append(_status_row(f"{label}_{col}_extrapolation", status, f"share>train_p99={_fmt(share_gt_p99)}; share>train_max={_fmt(share_gt_max)}"))
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "audit_feature_extrapolation.csv", index=False)
    return table, checks


def unit_orderflow_check(ow_eval: pd.DataFrame | None, proxy_rf_eval: pd.DataFrame | None, out_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    rows = []
    for label, df in [("OW_target_impact", ow_eval), ("fitted_proxy", proxy_rf_eval)]:
        if df is None or not {"orderFlow_market", "signed_volume"}.issubset(df.columns):
            rows.append({"strategy_name": label, "status": "SKIP", "reason": "required row-level evaluator columns unavailable"})
            checks.append(_status_row(f"{label}_orderflow_unit_check", "SKIP", "required evaluator columns unavailable"))
            continue
        market = pd.to_numeric(df["orderFlow_market"], errors="coerce").abs()
        q = pd.to_numeric(df["signed_volume"], errors="coerce").abs()
        denom = market.where(market > 0)
        ratio = q / denom
        share_gt10 = float((ratio > 10.0).mean(skipna=True))
        share_lt01 = float((ratio < 0.1).mean(skipna=True))
        status = "WARN" if share_gt10 > 0.05 else "PASS"
        rows.append(
            {
                "strategy_name": label,
                "status": status,
                "median_abs_orderFlow_market": float(market.median(skipna=True)),
                "p95_abs_orderFlow_market": float(market.quantile(0.95)),
                "median_abs_signed_volume": float(q.median(skipna=True)),
                "p95_abs_signed_volume": float(q.quantile(0.95)),
                "median_ratio_abs_strategy_to_market_orderFlow": float(ratio.median(skipna=True)),
                "p95_ratio_abs_strategy_to_market_orderFlow": float(ratio.quantile(0.95)),
                "share_strategy_gt_10x_market_orderFlow": share_gt10,
                "share_strategy_lt_0p1x_market_orderFlow": share_lt01,
                "share_zero_market_orderFlow": float((market == 0).mean()),
            }
        )
        checks.append(_status_row(f"{label}_orderflow_unit_check", status, f"share strategy >10x market orderFlow={_fmt(share_gt10)}"))
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "audit_unit_orderflow_vs_strategy_volume.csv", index=False)
    return table, checks


def timing_and_alpha_check(ow_df: pd.DataFrame | None, proxy_df: pd.DataFrame | None, out_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    rows = []
    for label, df in [("OW_target_impact", ow_df), ("fitted_proxy", proxy_df)]:
        if df is None:
            rows.append({"strategy_name": label, "status": "SKIP", "reason": "row-level strategy/evaluator unavailable"})
            checks.append(_status_row(f"{label}_timing", "SKIP", "row-level strategy/evaluator unavailable"))
            continue
        row: dict[str, Any] = {"strategy_name": label, "status": "PASS"}
        if {"gross_pnl", "position_before", "delta_mid"}.issubset(df.columns):
            expected = pd.to_numeric(df["position_before"], errors="coerce") * pd.to_numeric(df["delta_mid"], errors="coerce")
            err = (pd.to_numeric(df["gross_pnl"], errors="coerce") - expected).abs()
            row["gross_pnl_formula_max_error"] = float(err.max(skipna=True))
            if row["gross_pnl_formula_max_error"] > 1e-6:
                row["status"] = "FAIL"
        else:
            row["status"] = "SKIP"
            row["reason"] = "gross_pnl timing columns unavailable"
        if {"position_after", "delta_mid", "gross_pnl"}.issubset(df.columns):
            alt = pd.to_numeric(df["position_after"], errors="coerce") * pd.to_numeric(df["delta_mid"], errors="coerce")
            row["aggregate_difference_vs_position_after_pnl"] = float((alt - pd.to_numeric(df["gross_pnl"], errors="coerce")).sum(skipna=True))
        alpha_col = "alpha" if "alpha" in df.columns else "alpha_for_strategy" if "alpha_for_strategy" in df.columns else None
        if alpha_col is not None:
            alpha = pd.to_numeric(df[alpha_col], errors="coerce")
            if "future_return_h" in df.columns:
                row["corr_alpha_future_return_h"] = float(alpha.corr(pd.to_numeric(df["future_return_h"], errors="coerce")))
                row["corr_position_before_future_return_h"] = float(pd.to_numeric(df.get("position_before"), errors="coerce").corr(pd.to_numeric(df["future_return_h"], errors="coerce")))
            if "signed_volume" in df.columns:
                row["corr_trade_alpha"] = float(pd.to_numeric(df["signed_volume"], errors="coerce").corr(alpha))
            if "position_after" in df.columns:
                row["corr_position_after_alpha"] = float(pd.to_numeric(df["position_after"], errors="coerce").corr(alpha))
        rows.append(row)
        checks.append(_status_row(f"{label}_timing", row["status"], f"gross_pnl_error={_fmt(row.get('gross_pnl_formula_max_error', np.nan))}"))
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "audit_timing_and_alpha_alignment.csv", index=False)
    return table, checks


def cost_sign_check(ow_eval: pd.DataFrame | None, rf_eval: pd.DataFrame | None, proxy_rf_eval: pd.DataFrame | None, out_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    rows = []
    for label, df in [
        ("OW_target_impact__OW_transient", ow_eval),
        ("OW_target_impact__reduced_form", rf_eval),
        ("fitted_proxy__reduced_form", proxy_rf_eval),
    ]:
        if df is None or not {"signed_volume", "marginal_impact_bps", "fitted_impact_cost_signed", "fitted_impact_cost_abs"}.issubset(df.columns):
            rows.append({"strategy_evaluator": label, "status": "SKIP", "reason": "required cost columns unavailable"})
            checks.append(_status_row(f"{label}_cost_sign", "SKIP", "required cost columns unavailable"))
            continue
        qsign = np.sign(pd.to_numeric(df["signed_volume"], errors="coerce"))
        isign = np.sign(pd.to_numeric(df["marginal_impact_bps"], errors="coerce"))
        active = qsign != 0
        cost = pd.to_numeric(df["fitted_impact_cost_signed"], errors="coerce")
        abs_cost = pd.to_numeric(df["fitted_impact_cost_abs"], errors="coerce")
        share_same = float((qsign[active] == isign[active]).mean()) if active.any() else np.nan
        share_positive = float((cost[active] > 0).mean()) if active.any() else np.nan
        total_signed = float(cost.sum(skipna=True))
        total_abs = float(abs_cost.sum(skipna=True))
        netout = abs(total_signed) / total_abs if total_abs > 0 else np.nan
        status = "PASS" if np.isfinite(share_same) and share_same > 0.5 else "WARN"
        rows.append(
            {
                "strategy_evaluator": label,
                "status": status,
                "share_sign_trade_equals_sign_marginal_impact": share_same,
                "share_fitted_cost_positive": share_positive,
                "total_signed_cost": total_signed,
                "total_abs_cost": total_abs,
                "netout_ratio_abs_signed_over_abs_cost": netout,
            }
        )
        checks.append(_status_row(f"{label}_cost_sign", status, f"share_same_direction={_fmt(share_same)}; netout_ratio={_fmt(netout)}"))
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "audit_cost_sign_convention.csv", index=False)
    return table, checks


def _make_figures(
    fig_dir: Path,
    ow_df: pd.DataFrame | None,
    ow_eval: pd.DataFrame | None,
    rf_eval: pd.DataFrame | None,
    proxy_df: pd.DataFrame | None,
    proxy_rf_eval: pd.DataFrame | None,
    cost_table: pd.DataFrame,
    slope_table: pd.DataFrame,
    extrapolation_table: pd.DataFrame,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    if ow_df is not None and proxy_df is not None and "signed_volume" in ow_df.columns and "signed_volume" in proxy_df.columns:
        _hist_compare(
            pd.to_numeric(ow_df["signed_volume"], errors="coerce").abs(),
            pd.to_numeric(proxy_df["signed_volume"], errors="coerce").abs(),
            "OW target-impact",
            "fitted proxy",
            "OW vs fitted proxy trade size",
            "abs(signed volume)",
            fig_dir / "ow_vs_proxy_trade_size_histogram.png",
            log_x=True,
        )
    if ow_df is not None and proxy_df is not None and "participation_rate" in ow_df.columns and "participation_rate" in proxy_df.columns:
        _hist_compare(
            pd.to_numeric(ow_df["participation_rate"], errors="coerce"),
            pd.to_numeric(proxy_df["participation_rate"], errors="coerce"),
            "OW target-impact",
            "fitted proxy",
            "OW vs fitted proxy participation rate",
            "participation rate",
            fig_dir / "ow_vs_proxy_participation_histogram.png",
        )
    if len(cost_table) and "cost_bps_of_turnover" in cost_table.columns:
        plot_df = cost_table[cost_table.get("status", "PASS").eq("PASS")].copy()
        if len(plot_df):
            fig, ax = plt.subplots(figsize=(10, 5))
            labels = plot_df["strategy_name"].astype(str) + "\n" + plot_df["evaluator_name"].astype(str)
            ax.bar(labels, pd.to_numeric(plot_df["cost_bps_of_turnover"], errors="coerce"))
            ax.set_title("Cost bps of notional turnover comparison")
            ax.set_ylabel("bps of notional turnover")
            ax.tick_params(axis="x", labelrotation=35)
            _save_fig(fig, fig_dir / "cost_bps_of_turnover_comparison.png")
    if proxy_df is not None and "impact_slope_bps_per_share" in proxy_df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = pd.to_numeric(proxy_df["impact_slope_bps_per_share"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(x):
            ax.hist(x[x > 0], bins=80)
            ax.set_xscale("log")
        ax.set_title("Proxy impact slope distribution")
        ax.set_xlabel("impact slope bps/share")
        ax.set_ylabel("Rows")
        _save_fig(fig, fig_dir / "proxy_slope_distribution.png")
    if proxy_df is not None and proxy_rf_eval is not None:
        internal = _portfolio_cumsum(proxy_df, "fitted_impact_cost")
        full = _portfolio_cumsum(proxy_rf_eval, "fitted_impact_cost_signed")
        fig, ax = plt.subplots(figsize=(10, 5))
        if len(internal):
            ax.plot(internal["datetime"], internal["cum_fitted_impact_cost"], label="proxy internal cost")
        if len(full):
            ax.plot(full["datetime"], full["cum_fitted_impact_cost_signed"], label="full RF evaluator cost")
        ax.set_title("Proxy internal vs full RF cost cumulative")
        ax.set_ylabel("Cumulative cost")
        ax.legend()
        _save_fig(fig, fig_dir / "proxy_internal_vs_full_rf_cost_cumulative.png")
    if len(extrapolation_table):
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_df = extrapolation_table[extrapolation_table["feature"].eq("x_flow")].copy()
        if len(plot_df):
            labels = plot_df["strategy_name"].astype(str)
            ax.bar(labels, pd.to_numeric(plot_df["share_abs_scenario_gt_train_p99"], errors="coerce"))
        ax.set_title("x_flow feature extrapolation")
        ax.set_ylabel("Share abs(scenario x_flow) > train p99")
        _save_fig(fig, fig_dir / "ow_vs_proxy_x_flow_extrapolation.png")
    if rf_eval is not None and proxy_rf_eval is not None:
        ow_ts = _portfolio_cumsum(rf_eval, "net_pnl_fitted_model")
        px_ts = _portfolio_cumsum(proxy_rf_eval, "net_pnl_fitted_model")
        fig, ax = plt.subplots(figsize=(10, 5))
        if len(ow_ts):
            ax.plot(ow_ts["datetime"], ow_ts["cum_net_pnl_fitted_model"], label="OW under RF evaluator")
        if len(px_ts):
            ax.plot(px_ts["datetime"], px_ts["cum_net_pnl_fitted_model"], label="proxy under RF evaluator")
        ax.set_title("OW vs proxy cumulative wealth under same RF evaluator")
        ax.set_ylabel("Cumulative net PnL")
        ax.legend()
        _save_fig(fig, fig_dir / "ow_vs_proxy_cumulative_wealth_under_same_rf_evaluator.png")
    sample = proxy_df if proxy_df is not None else ow_df
    if sample is not None and {"stock", "trading_date", "datetime", "signed_volume"}.issubset(sample.columns):
        s = _ensure_timestamp(sample)
        key = (
            s.assign(abs_q=pd.to_numeric(s["signed_volume"], errors="coerce").abs())
            .groupby(["stock", "trading_date"], as_index=False)["abs_q"]
            .sum()
            .sort_values("abs_q", ascending=False)
            .head(1)
        )
        if len(key):
            stock, day = key.iloc[0]["stock"], key.iloc[0]["trading_date"]
            path_df = s[(s["stock"].eq(stock)) & (s["trading_date"].astype(str).eq(str(day)))].copy()
            fig, ax1 = plt.subplots(figsize=(11, 5))
            alpha_col = "alpha" if "alpha" in path_df.columns else "alpha_for_strategy" if "alpha_for_strategy" in path_df.columns else None
            if alpha_col:
                ax1.plot(path_df["datetime"], path_df[alpha_col], label="alpha", color="tab:blue")
            if "position_after" in path_df.columns:
                ax2 = ax1.twinx()
                ax2.plot(path_df["datetime"], path_df["position_after"], label="position", color="tab:orange", alpha=0.8)
                ax2.bar(path_df["datetime"], path_df["signed_volume"], label="trade", color="tab:green", alpha=0.25, width=0.002)
                ax2.set_ylabel("Position / trade")
            ax1.set_title(f"Alpha-position-trade sample: {stock} {day}")
            ax1.set_ylabel("Alpha")
            _save_fig(fig, fig_dir / "alpha_position_trade_alignment_sample.png")


def _write_report(
    report_md: Path,
    checks: pd.DataFrame,
    trade_table: pd.DataFrame,
    cost_table: pd.DataFrame,
    proxy_compare: pd.DataFrame,
    unit_table: pd.DataFrame,
    extrapolation_table: pd.DataFrame,
    save_trades: bool | None,
) -> None:
    status_counts = checks["status"].value_counts().to_dict() if len(checks) else {}
    worst = "PASS"
    if "FAIL" in status_counts:
        worst = "FAIL"
    elif "WARN" in status_counts:
        worst = "WARN"
    ow_proxy_ratio = np.nan
    if len(trade_table) and (trade_table["strategy_name"] == "OW_to_proxy_ratio").any():
        ow_proxy_ratio = float(trade_table.loc[trade_table["strategy_name"].eq("OW_to_proxy_ratio"), "total_abs_volume"].iloc[0])
    proxy_ratio = np.nan
    if len(proxy_compare) and "ratio_full_rf_to_proxy_internal" in proxy_compare.columns:
        proxy_ratio = float(proxy_compare["ratio_full_rf_to_proxy_internal"].iloc[0])
    high_cost_explained = (
        "OW trades materially more than the proxy, so higher fitted costs are consistent with turnover."
        if np.isfinite(ow_proxy_ratio) and ow_proxy_ratio > 2
        else "OW/proxy turnover ratio is not enough by itself to explain cost differences."
    )
    if not np.isfinite(proxy_ratio):
        proxy_cost_note = "Proxy internal cost versus full RF evaluator was not computed in this run."
    elif 0.2 <= proxy_ratio <= 5:
        proxy_cost_note = "Proxy internal cost and full reduced-form evaluator cost are in a comparable range."
    else:
        proxy_cost_note = "Proxy internal cost differs materially from the full RF evaluator; use the full evaluator for final economic comparison."
    lines = [
        "# Strategy Implementation Audit",
        "",
        "## 1. Executive Summary",
        f"- Overall status: {worst}",
        f"- Check counts: {status_counts}",
        f"- Row-level OW trade file saved: {bool(save_trades)}",
        f"- OW/proxy total volume ratio: {_fmt(ow_proxy_ratio)}",
        f"- Full RF cost on proxy / proxy internal cost: {_fmt(proxy_ratio)}",
        f"- Interpretation: {high_cost_explained}",
        f"- Proxy cost interpretation: {proxy_cost_note}",
        "",
        "## 2. Trade Size Comparison",
        "```text",
        trade_table.to_string(index=False) if len(trade_table) else "No trade-size table available.",
        "```",
        "",
        "## 3. Cost Per Turnover",
        "```text",
        cost_table.to_string(index=False) if len(cost_table) else "No cost table available.",
        "```",
        "",
        "## 4. Proxy Cost And Slope Validation",
        "```text",
        proxy_compare.to_string(index=False) if len(proxy_compare) else "No proxy comparison available.",
        "```",
        "",
        "## 5. Feature Extrapolation",
        "```text",
        extrapolation_table.to_string(index=False) if len(extrapolation_table) else "No extrapolation table available.",
        "```",
        "",
        "## 6. Unit Checks",
        "```text",
        unit_table.to_string(index=False) if len(unit_table) else "No unit check table available.",
        "```",
        "",
        "## 7. Timing And Sign Checks",
        "```text",
        checks.to_string(index=False) if len(checks) else "No checks available.",
        "```",
        "",
        "## 8. Conclusions",
        "- OW target-impact mechanics are considered mechanically correct when timing, position dynamics, and cost-sign checks pass.",
        "- High fitted costs are not automatically a bug; they can arise from aggressive turnover or scenario features outside the fitted training domain.",
        "- The fitted proxy is a local myopic quadratic-cost strategy. Trust final report tables more when proxy internal costs are close to full reduced-form evaluator costs on the same proxy trades.",
        "- If row-level OW trades were not saved, rerun with `--save-trades --run-strategy-audit` for the strictest implementation audit.",
        "",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    report_md.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")


def run_strategy_implementation_audit(
    experiment_path: str | Path,
    pair_id: int,
    *,
    use_saved_data_only: bool = False,
) -> dict[str, Path]:
    """Run the strategy implementation audit for one experiment and pair."""

    context = _load_context(Path(experiment_path), pair_id)
    exp = context["experiment_path"]
    out_dir = exp / "audit_tables"
    fig_dir = exp / "audit_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ow_trades = context["ow_trades"]
    ow_eval = context["ow_eval"]
    rf_eval = context["rf_eval"]
    proxy_trades = context["proxy_trades"]
    metadata = context["metadata"]
    save_trades = metadata.get("save_trades")

    # Use RF evaluator rows as the OW trade proxy for sizing and unit checks when
    # --save-trades was omitted. Strict OW internal timing checks remain SKIP.
    ow_trade_like = ow_trades if ow_trades is not None else rf_eval
    checks: list[dict[str, Any]] = []
    if ow_trades is None:
        checks.append(
            _status_row(
                "ow_row_level_trade_file",
                "SKIP",
                "my_ow_trades.csv not saved; using evaluator rows for trade-size/unit diagnostics only",
            )
        )
    else:
        checks.append(_status_row("ow_row_level_trade_file", "PASS", f"rows={len(ow_trades)}"))

    train_raw = test_raw = rf_params = None
    proxy_rf_eval = None
    try:
        _, train_raw, test_raw, rf_params = _load_pair_raw_data(context, pair_id)
        if proxy_trades is not None and not use_saved_data_only:
            proxy_rf_eval = evaluate_marginal_impact_from_strategy_trades(
                train_raw,
                test_raw,
                proxy_trades,
                rf_params,
                pair_id,
                model_name="reduced_form",
            )
            # Keep the row-level audit evaluator because it is the key comparison.
            proxy_rf_eval.to_csv(out_dir / "audit_proxy_full_rf_evaluator_rows.csv", index=False)
            checks.append(_status_row("proxy_full_rf_evaluator_available", "PASS", f"rows={len(proxy_rf_eval)}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(_status_row("raw_data_or_proxy_full_rf_evaluator", "WARN", f"could not load/recompute raw evaluator data: {exc}"))

    trade_table, trade_checks = trade_size_comparison(ow_trade_like, proxy_trades, out_dir)
    checks.extend(trade_checks)
    recon_table, slope_table, proxy_checks = proxy_cost_consistency(proxy_trades, out_dir)
    checks.extend(proxy_checks)
    proxy_compare, proxy_compare_checks = proxy_full_rf_comparison(proxy_trades, proxy_rf_eval, out_dir)
    checks.extend(proxy_compare_checks)
    cost_table = cost_per_turnover_table(ow_eval, rf_eval, proxy_trades, proxy_rf_eval, out_dir)
    extrapolation_table, extrapolation_checks = feature_extrapolation(train_raw, test_raw, ow_trade_like, proxy_trades, out_dir)
    checks.extend(extrapolation_checks)
    unit_table, unit_checks = unit_orderflow_check(rf_eval, proxy_rf_eval, out_dir)
    checks.extend(unit_checks)
    timing_table, timing_checks = timing_and_alpha_check(ow_trades, proxy_trades, out_dir)
    checks.extend(timing_checks)
    sign_table, sign_checks = cost_sign_check(ow_eval, rf_eval, proxy_rf_eval, out_dir)
    checks.extend(sign_checks)

    # Extra warning: proxy internal costs that are tiny relative to full RF costs
    # should be interpreted with the full evaluator, not the local proxy alone.
    if len(proxy_compare) and "ratio_full_rf_to_proxy_internal" in proxy_compare.columns:
        ratio = float(proxy_compare["ratio_full_rf_to_proxy_internal"].iloc[0])
        if np.isfinite(ratio) and ratio > 5:
            checks.append(_status_row("proxy_cost_too_low_vs_full_rf", "WARN", f"full RF/proxy internal cost ratio={_fmt(ratio)}"))
        elif np.isfinite(ratio):
            checks.append(_status_row("proxy_cost_vs_full_rf", "PASS", f"full RF/proxy internal cost ratio={_fmt(ratio)}"))

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(out_dir / "audit_strategy_implementation_checks.csv", index=False)
    timing_table.to_csv(out_dir / "audit_timing_and_alpha_alignment.csv", index=False)
    sign_table.to_csv(out_dir / "audit_cost_sign_convention.csv", index=False)

    _make_figures(
        fig_dir,
        ow_trade_like,
        ow_eval,
        rf_eval,
        proxy_trades,
        proxy_rf_eval,
        cost_table,
        slope_table,
        extrapolation_table,
    )
    report_md = exp / "audit_strategy_implementation_report.md"
    _write_report(
        report_md,
        checks_df,
        trade_table,
        cost_table,
        proxy_compare,
        unit_table,
        extrapolation_table,
        save_trades,
    )
    return {
        "report": report_md,
        "checks": out_dir / "audit_strategy_implementation_checks.csv",
        "tables": out_dir,
        "figures": fig_dir,
    }


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Audit integrated strategy implementations")
    parser.add_argument("--experiment-path", required=True)
    parser.add_argument("--pair-id", type=int, required=True)
    parser.add_argument("--use-saved-data-only", action="store_true")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    paths = run_strategy_implementation_audit(
        args.experiment_path,
        args.pair_id,
        use_saved_data_only=args.use_saved_data_only,
    )
    print(f"Audit report: {paths['report']}")
    print(f"Audit checks: {paths['checks']}")
    print(f"Audit tables: {paths['tables']}")
    print(f"Audit figures: {paths['figures']}")


if __name__ == "__main__":
    main()
