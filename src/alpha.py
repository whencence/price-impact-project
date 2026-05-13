"""Clock-time synthetic alpha pipeline for project section 2.4.

The final convention is clock time, not row count:

    alpha_raw_t^h = x r_t^h + y DeltaW_t^h / P_t

where h is measured in minutes, P_t = mid_t, and DeltaW_t^h ~ Normal(0, h).
Future prices are found within the same stock/date by taking the first
available timestamp at or after t + h, subject to a tolerance.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.config import AlphaConfig


MIN_VALID_OBS = 2
BASELINE_ALPHA_CLOCK_FILENAME = "synthetic_alpha_baseline_h5m_rho010.csv"
BASELINE_ALPHA_COMPAT_FILENAME = "synthetic_alpha_baseline_h5_rho010.csv"
DIAGNOSTICS_FILENAME = "synthetic_alpha_diagnostics.csv"


def _warn(message: str) -> None:
    print(f"WARNING: {message}")


def _scenario_seed(random_seed: int, horizon_minutes: float, target_corr: float) -> int:
    return int(random_seed + round(10_000 * horizon_minutes) + round(1_000_000 * target_corr))


def _format_minutes(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def make_alpha_scenario_key(h: int | float, rho: float) -> str:
    """Return a compact scenario key such as h5m_rho010."""

    return f"h{_format_minutes(float(h))}m_rho{int(round(rho * 100)):03d}"


def _state_col_name(half_life_minutes: float) -> str:
    return f"alpha_state_H{_format_minutes(half_life_minutes)}m"


def parse_time_to_timedelta(time_series: pd.Series) -> pd.Series:
    """Parse HH:MM:SS, HH:MM:SS.sss, HHMMSS, seconds, ms, or us to timedelta."""

    as_string = time_series.astype("string").str.strip()
    parsed = pd.to_timedelta(as_string, errors="coerce")
    numeric = pd.to_numeric(time_series, errors="coerce")
    unresolved = parsed.isna() & numeric.notna()
    if unresolved.any():
        nums = numeric.loc[unresolved]
        max_abs = float(nums.abs().max())
        if max_abs <= 24 * 60 * 60:
            parsed.loc[unresolved] = pd.to_timedelta(nums, unit="s")
        elif max_abs <= 24 * 60 * 60 * 1_000:
            parsed.loc[unresolved] = pd.to_timedelta(nums, unit="ms")
        elif max_abs <= 24 * 60 * 60 * 1_000_000:
            parsed.loc[unresolved] = pd.to_timedelta(nums, unit="us")
        else:
            padded = nums.astype("Int64").astype("string").str.zfill(6)
            hh = pd.to_numeric(padded.str.slice(0, 2), errors="coerce")
            mm = pd.to_numeric(padded.str.slice(2, 4), errors="coerce")
            ss = pd.to_numeric(padded.str.slice(4, 6), errors="coerce")
            parsed.loc[unresolved] = (
                pd.to_timedelta(hh, unit="h")
                + pd.to_timedelta(mm, unit="m")
                + pd.to_timedelta(ss, unit="s")
            )
    return parsed


def _parse_date_series(date_series: pd.Series) -> pd.Series:
    as_string = date_series.astype("string").str.strip()
    yyyymmdd = as_string.str.fullmatch(r"\d{8}", na=False)
    parsed_default = pd.to_datetime(as_string.where(~yyyymmdd), errors="coerce")
    parsed_numeric = pd.to_datetime(as_string.where(yyyymmdd), format="%Y%m%d", errors="coerce")
    return parsed_default.fillna(parsed_numeric)


def add_timestamp_column(df: pd.DataFrame, config: AlphaConfig) -> pd.DataFrame:
    """Return a copy with parsed date_parsed and timestamp columns."""

    missing = [col for col in [config.date_col, config.time_col] if col not in df.columns]
    if missing:
        raise ValueError(f"cannot create timestamp; missing columns: {missing}")
    out = df.copy()
    out["date_parsed"] = _parse_date_series(out[config.date_col])
    if pd.api.types.is_datetime64_any_dtype(out[config.time_col]):
        out[config.timestamp_col] = pd.to_datetime(out[config.time_col], errors="coerce")
    else:
        time_delta = parse_time_to_timedelta(out[config.time_col])
        out[config.timestamp_col] = out["date_parsed"] + time_delta
    failure_rate = float(out[config.timestamp_col].isna().mean()) if len(out) else 1.0
    if failure_rate > 0:
        _warn(f"timestamp parse failure rate: {failure_rate:.4%}")
    if failure_rate > 0.05:
        raise ValueError("timestamp parsing failed for more than 5% of rows")
    return out


def validate_bin_sample_data(df: pd.DataFrame, config: AlphaConfig) -> None:
    """Validate binSamples data before clock-time alpha construction."""

    if df.empty:
        raise ValueError("binSamples DataFrame must not be empty")
    required = [config.date_col, config.time_col, config.stock_col, config.price_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"binSamples DataFrame is missing required columns: {missing}")
    price = pd.to_numeric(df[config.price_col], errors="coerce")
    missing_price = int(price.isna().sum())
    non_positive = int((price <= 0).fillna(False).sum())
    duplicate_keys = int(df.duplicated([config.stock_col, config.date_col, config.time_col]).sum())
    if missing_price:
        _warn(f"{missing_price} rows have missing/non-numeric {config.price_col}")
    if non_positive:
        _warn(f"{non_positive} rows have non-positive {config.price_col}")
    if duplicate_keys:
        _warn(f"{duplicate_keys} duplicate stock/date/time keys")
    if int((price > 0).sum()) == 0:
        raise ValueError(f"no positive {config.price_col} values found")


def prepare_alpha_base_data(df: pd.DataFrame, config: AlphaConfig) -> pd.DataFrame:
    """Prepare and sort binSamples once for clock-time alpha generation."""

    validate_bin_sample_data(df, config)
    useful = [
        config.date_col,
        config.time_col,
        config.stock_col,
        config.price_col,
        "midEnd",
        "spread",
        "depth",
        "volume",
        "source_file",
    ]
    out = df.loc[:, [col for col in useful if col in df.columns]].copy()
    out[config.price_col] = pd.to_numeric(out[config.price_col], errors="coerce")
    out = add_timestamp_column(out, config)
    out["usable_price"] = out[config.price_col].notna() & (out[config.price_col] > 0)
    out = out.sort_values([config.stock_col, config.date_col, config.timestamp_col]).reset_index(drop=True)
    out["row_id_within_stock_date"] = out.groupby(
        [config.stock_col, config.date_col], sort=False
    ).cumcount()
    out["dt_seconds"] = (
        out.groupby([config.stock_col, config.date_col], sort=False)[config.timestamp_col]
        .diff()
        .dt.total_seconds()
    )
    gaps = out["dt_seconds"].dropna()
    if not gaps.empty:
        median_gap = float(gaps.median())
        p95_gap = float(gaps.quantile(0.95))
        max_gap = float(gaps.max())
        print(
            f"Time-gap diagnostics: median={median_gap:.2f}s, "
            f"p95={p95_gap:.2f}s, max={max_gap:.2f}s"
        )
        if abs(median_gap - 10.0) > 2.0 or p95_gap > 15.0:
            _warn("binSamples are not perfectly regular; clock-time horizons are being used")
    return out


# Backward-compatible alias from earlier implementation.
prepare_base_alpha_data = prepare_alpha_base_data


def add_clock_time_future_returns(base_df: pd.DataFrame, config: AlphaConfig) -> pd.DataFrame:
    """Add clock-time future returns using first timestamp >= t+h within stock/date."""

    out = base_df.copy()
    horizon_delta = pd.to_timedelta(config.horizon_minutes, unit="m")
    out["target_timestamp_h"] = out[config.timestamp_col] + horizon_delta
    out["future_timestamp_h"] = pd.NaT
    out["future_mid_h"] = np.nan
    out["future_time_gap_seconds"] = np.nan

    for _, idx in out.groupby([config.stock_col, config.date_col], sort=False).groups.items():
        positions = np.asarray(idx)
        timestamps = out.loc[positions, config.timestamp_col].to_numpy(dtype="datetime64[ns]")
        targets = out.loc[positions, "target_timestamp_h"].to_numpy(dtype="datetime64[ns]")
        mids = out.loc[positions, config.price_col].to_numpy(dtype=float)
        future_pos = np.searchsorted(timestamps, targets, side="left")
        has_future = future_pos < len(positions)
        dest = positions[has_future]
        src = positions[future_pos[has_future]]
        out.loc[dest, "future_timestamp_h"] = out.loc[src, config.timestamp_col].to_numpy()
        out.loc[dest, "future_mid_h"] = mids[future_pos[has_future]]

    out["future_time_gap_seconds"] = (
        out["future_timestamp_h"] - out["target_timestamp_h"]
    ).dt.total_seconds()
    out["future_return_h"] = out["future_mid_h"] / out[config.price_col] - 1.0
    out["valid_future_return"] = (
        out[config.price_col].gt(0)
        & out["future_mid_h"].gt(0)
        & out["future_return_h"].notna()
        & out["future_time_gap_seconds"].ge(0)
        & out["future_time_gap_seconds"].le(config.future_tolerance_seconds)
    )
    out["horizon_minutes"] = float(config.horizon_minutes)
    _future_return_warnings(out, config)
    return out


def _future_return_warnings(df: pd.DataFrame, config: AlphaConfig) -> None:
    valid = df.loc[df["valid_future_return"]].copy()
    missing_fraction = 1.0 - len(valid) / len(df) if len(df) else 1.0
    if missing_fraction > 0.20:
        _warn(f"{missing_fraction:.2%} future returns invalid for h={config.horizon_minutes:g}m")
    if not valid.empty:
        median_gap = float(valid["future_time_gap_seconds"].median())
        if median_gap > 0.5 * config.future_tolerance_seconds:
            _warn("median future lookup gap is large relative to tolerance")
    # Same stock/date is enforced by per-group searchsorted. This check catches
    # accidental negative gaps or malformed timestamps.
    if (df["future_time_gap_seconds"].dropna() < 0).any():
        raise RuntimeError("future return matching produced negative time gaps")


def estimate_global_alpha_moments(alpha_df: pd.DataFrame) -> dict[str, float | int]:
    """Estimate Var(r_t^h) and E[1/P_t^2] from valid clock-time returns."""

    required = {"valid_future_return", "future_return_h", "mid"}
    missing = required.difference(alpha_df.columns)
    if missing:
        raise ValueError(f"cannot estimate moments; missing columns: {sorted(missing)}")
    valid = alpha_df.loc[alpha_df["valid_future_return"] & alpha_df["mid"].gt(0)].copy()
    n_valid = int(len(valid))
    if n_valid < MIN_VALID_OBS:
        raise ValueError(f"need at least {MIN_VALID_OBS} valid rows to estimate alpha moments")
    var_future_return = float(valid["future_return_h"].var(ddof=1))
    mean_inv_price_sq = float((1.0 / valid["mid"] ** 2).mean())
    if var_future_return <= 0:
        raise ValueError("var_future_return must be positive")
    if mean_inv_price_sq <= 0:
        raise ValueError("mean_inv_price_sq must be positive")
    return {
        "var_future_return": var_future_return,
        "mean_inv_price_sq": mean_inv_price_sq,
        "n_valid": n_valid,
    }


def compute_xy_from_target_corr(
    var_future_return: float,
    mean_inv_price_sq: float,
    horizon_minutes: float,
    target_corr: float,
) -> tuple[float, float]:
    """Compute x/y coefficients using clock-time horizon in minutes."""

    if not 0.0 < target_corr < 1.0:
        raise ValueError("target_corr must be strictly between 0 and 1")
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive")
    if var_future_return <= 0:
        raise ValueError("var_future_return must be positive")
    if mean_inv_price_sq <= 0:
        raise ValueError("mean_inv_price_sq must be positive")
    x = target_corr**2
    y = target_corr * np.sqrt(1.0 - target_corr**2) * np.sqrt(
        var_future_return / (mean_inv_price_sq * horizon_minutes)
    )
    return float(x), float(y)


def _diagnostics_from_valid_arrays(
    horizon_df: pd.DataFrame,
    config: AlphaConfig,
    target_corr: float,
    x: float,
    y: float,
    alpha_valid: np.ndarray,
) -> dict[str, float | int | bool]:
    valid_mask = horizon_df["valid_future_return"].to_numpy(dtype=bool)
    future_return = horizon_df.loc[valid_mask, "future_return_h"].to_numpy(dtype=float)
    mid = horizon_df.loc[valid_mask, config.price_col].to_numpy(dtype=float)
    gaps = horizon_df.loc[valid_mask, "future_time_gap_seconds"]
    n_total = int(len(horizon_df))
    n_valid = int(valid_mask.sum())
    missing = 1.0 - n_valid / n_total if n_total else 1.0
    if n_valid >= MIN_VALID_OBS:
        var_r = float(np.var(future_return, ddof=1))
        mean_inv = float(np.mean(1.0 / mid**2))
        alpha_mean = float(np.mean(alpha_valid))
        alpha_std = float(np.std(alpha_valid, ddof=1))
        ret_mean = float(np.mean(future_return))
        ret_std = float(np.std(future_return, ddof=1))
        cov = float(np.cov(alpha_valid, future_return, ddof=1)[0, 1])
        corr = cov / (alpha_std * ret_std) if alpha_std > 0 and ret_std > 0 else np.nan
        slope = cov / float(np.var(alpha_valid, ddof=1)) if alpha_std > 0 else np.nan
    else:
        var_r = mean_inv = alpha_mean = alpha_std = ret_mean = ret_std = corr = slope = np.nan
    return {
        "horizon_minutes": float(config.horizon_minutes),
        "target_corr": float(target_corr),
        "n_rows_total": n_total,
        "n_valid_future_return": n_valid,
        "fraction_missing_future_return": float(missing),
        "median_future_time_gap_seconds": float(gaps.median()) if len(gaps) else np.nan,
        "p95_future_time_gap_seconds": float(gaps.quantile(0.95)) if len(gaps) else np.nan,
        "max_future_time_gap_seconds": float(gaps.max()) if len(gaps) else np.nan,
        "var_future_return": var_r,
        "mean_inv_price_sq": mean_inv,
        "alpha_x": float(x),
        "alpha_y": float(y),
        "empirical_corr_alpha_return": corr,
        "alpha_mean": alpha_mean,
        "alpha_std": alpha_std,
        "future_return_mean": ret_mean,
        "future_return_std": ret_std,
        "realized_unbiased_slope": slope,
        "corr_abs_error": float(abs(corr - target_corr)) if np.isfinite(corr) else np.nan,
    }


def add_alpha_for_rho(
    horizon_df: pd.DataFrame,
    target_corr: float,
    moments: dict[str, float | int],
    random_seed: int,
    config: AlphaConfig,
    keep_full_output: bool,
) -> tuple[pd.DataFrame | None, dict[str, float | int | bool]]:
    """Compute raw alpha for one rho; keep full row output only when requested."""

    x, y = compute_xy_from_target_corr(
        float(moments["var_future_return"]),
        float(moments["mean_inv_price_sq"]),
        config.horizon_minutes,
        target_corr,
    )
    rng = np.random.default_rng(random_seed)
    delta_w = np.sqrt(config.horizon_minutes) * rng.standard_normal(len(horizon_df))
    valid = horizon_df["valid_future_return"].to_numpy(dtype=bool)
    future_return = horizon_df.loc[valid, "future_return_h"].to_numpy(dtype=float)
    mid = horizon_df.loc[valid, config.price_col].to_numpy(dtype=float)
    alpha_valid = x * future_return + y * delta_w[valid] / mid
    diagnostics = _diagnostics_from_valid_arrays(horizon_df, config, target_corr, x, y, alpha_valid)

    if not keep_full_output:
        return None, diagnostics

    alpha_df = horizon_df.copy()
    alpha_df["delta_w"] = delta_w
    alpha_df["alpha_synthetic"] = 0.0
    alpha_df.loc[valid, "alpha_synthetic"] = alpha_valid
    if config.alpha_clip is not None:
        alpha_df["alpha_synthetic"] = alpha_df["alpha_synthetic"].clip(
            -config.alpha_clip, config.alpha_clip
        )
    alpha_df["alpha_x"] = x
    alpha_df["alpha_y"] = y
    alpha_df["target_corr"] = float(target_corr)
    alpha_df["horizon_minutes"] = float(config.horizon_minutes)
    return alpha_df, diagnostics


def build_synthetic_alpha_clock_time(df: pd.DataFrame, config: AlphaConfig) -> pd.DataFrame:
    """Build a full row-level raw synthetic alpha for one clock-time scenario."""

    base = prepare_alpha_base_data(df, config)
    horizon_df = add_clock_time_future_returns(base, config)
    moments = estimate_global_alpha_moments(horizon_df)
    alpha_df, _ = add_alpha_for_rho(
        horizon_df,
        config.target_corr,
        moments,
        config.random_seed,
        config,
        keep_full_output=True,
    )
    if alpha_df is None:
        raise RuntimeError("failed to construct alpha output")
    return alpha_df


def add_alpha_decay_state_clock_time(
    alpha_df: pd.DataFrame,
    alpha_col: str = "alpha_synthetic",
    half_life_minutes: float = 5.0,
    config: AlphaConfig | None = None,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Add irregular-clock-time exponential alpha state for one half-life.

    First state in each stock/date group equals the raw alpha at that timestamp.
    Subsequent states use phi_t = exp(-ln(2) dt_minutes / H).
    """

    if half_life_minutes <= 0:
        raise ValueError("half_life_minutes must be positive")
    config = config or AlphaConfig()
    output_col = output_col or _state_col_name(half_life_minutes)
    out = alpha_df.copy().sort_values(
        [config.stock_col, config.date_col, config.timestamp_col]
    ).reset_index(drop=True)
    states = np.zeros(len(out), dtype=float)
    raw = pd.to_numeric(out[alpha_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    for _, idx in out.groupby([config.stock_col, config.date_col], sort=False).groups.items():
        positions = np.asarray(idx)
        if len(positions) == 0:
            continue
        states[positions[0]] = raw[positions[0]]
        times = out.loc[positions, config.timestamp_col]
        dt_minutes = times.diff().dt.total_seconds().fillna(0.0).to_numpy(dtype=float) / 60.0
        for j in range(1, len(positions)):
            phi = np.exp(-np.log(2.0) * max(dt_minutes[j], 0.0) / half_life_minutes)
            states[positions[j]] = phi * states[positions[j - 1]] + (1.0 - phi) * raw[positions[j]]
    out[output_col] = states
    if out[output_col].isna().any():
        raise RuntimeError(f"{output_col} contains NaNs")
    return out


def add_alpha_decay_state_grid_clock_time(
    alpha_df: pd.DataFrame,
    half_lives_minutes: list[float] | tuple[float, ...] = (1.0, 5.0, 30.0, 60.0),
    alpha_col: str = "alpha_synthetic",
    config: AlphaConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add alpha state columns for all requested clock-time half-lives."""

    config = config or AlphaConfig()
    out = alpha_df.copy()
    metadata = []
    for half_life in half_lives_minutes:
        output_col = _state_col_name(float(half_life))
        out = add_alpha_decay_state_clock_time(out, alpha_col, float(half_life), config, output_col)
        metadata.append({"half_life_minutes": float(half_life), "output_col": output_col})
    return out, pd.DataFrame(metadata)


def alpha_decay_state_diagnostics_clock_time(
    alpha_df: pd.DataFrame, metadata_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute raw-vs-decayed alpha state diagnostics by half-life."""

    valid = alpha_df.loc[alpha_df["valid_future_return"]].copy()
    rows = []
    raw = valid["alpha_synthetic"].fillna(0.0)
    future_return = valid["future_return_h"]
    raw_std = float(raw.std(ddof=1)) if len(valid) > 1 else np.nan
    raw_abs = float(raw.abs().mean()) if len(valid) else np.nan
    for _, meta in metadata_df.iterrows():
        col = str(meta["output_col"])
        state = valid[col].fillna(0.0)
        state_std = float(state.std(ddof=1)) if len(valid) > 1 else np.nan
        state_abs = float(state.abs().mean()) if len(valid) else np.nan
        rows.append(
            {
                "half_life_minutes": float(meta["half_life_minutes"]),
                "output_col": col,
                "n_valid": int(len(valid)),
                "corr_raw_alpha_future_return": float(raw.corr(future_return)) if len(valid) > 1 else np.nan,
                "corr_alpha_state_future_return": float(state.corr(future_return)) if len(valid) > 1 else np.nan,
                "raw_alpha_std": raw_std,
                "alpha_state_std": state_std,
                "raw_alpha_std_bps": raw_std * 10_000.0 if np.isfinite(raw_std) else np.nan,
                "alpha_state_std_bps": state_std * 10_000.0 if np.isfinite(state_std) else np.nan,
                "state_to_raw_std_ratio": state_std / raw_std if raw_std and raw_std > 0 else np.nan,
                "mean_abs_raw_alpha": raw_abs,
                "mean_abs_alpha_state": state_abs,
                "state_to_raw_mean_abs_ratio": state_abs / raw_abs if raw_abs and raw_abs > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _emit_diagnostic_warnings(row: dict[str, Any]) -> None:
    label = f"h={row['horizon_minutes']:g}m, rho={row['target_corr']:.2f}"
    if row["fraction_missing_future_return"] > 0.20:
        _warn(f"{label}: high missing future return fraction")
    if np.isfinite(row["corr_abs_error"]) and row["corr_abs_error"] > 0.03:
        _warn(f"{label}: empirical correlation differs from target by {row['corr_abs_error']:.4f}")
    slope = row["realized_unbiased_slope"]
    if np.isfinite(slope) and abs(slope - 1.0) > 0.10:
        _warn(f"{label}: realized_unbiased_slope={slope:.4f}, far from 1")


def _baseline_columns(alpha_df: pd.DataFrame, config: AlphaConfig) -> list[str]:
    wanted = [
        config.date_col,
        config.time_col,
        config.stock_col,
        config.price_col,
        "midEnd",
        "spread",
        "depth",
        config.timestamp_col,
        "target_timestamp_h",
        "future_timestamp_h",
        "future_mid_h",
        "future_time_gap_seconds",
        "future_return_h",
        "alpha_synthetic",
        "baseline_alpha_for_strategy",
        "alpha_x",
        "alpha_y",
        "target_corr",
        "horizon_minutes",
        "valid_future_return",
        "delta_w",
        "source_file",
    ]
    state_cols = [col for col in alpha_df.columns if col.startswith("alpha_state_H")]
    return [col for col in wanted + state_cols if col in alpha_df.columns]


def run_synthetic_alpha_grid_clock_time(
    bin_df: pd.DataFrame,
    horizons_minutes: list[float] | tuple[float, ...] = (1.0, 5.0, 10.0),
    rhos: list[float] | tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50),
    baseline_horizon_minutes: float = 5.0,
    baseline_rho: float = 0.10,
    decay_half_lives_minutes: list[float] | tuple[float, ...] = (1.0, 5.0, 30.0, 60.0),
    baseline_decay_half_life_minutes: float = 5.0,
    output_dir: Path = Path("outputs") / "alphas",
    random_seed: int = 42,
    save_all_alpha_outputs: bool = False,
    make_plots: bool = True,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Run the final clock-time synthetic alpha grid."""

    total_start = time.perf_counter()
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    base_config = AlphaConfig(
        horizon_minutes=baseline_horizon_minutes,
        target_corr=baseline_rho,
        random_seed=random_seed,
        alpha_decay_half_life_minutes=baseline_decay_half_life_minutes,
        alpha_decay_half_life_grid_minutes=tuple(decay_half_lives_minutes),
    )
    prep_start = time.perf_counter()
    base_df = prepare_alpha_base_data(bin_df, base_config)
    print(f"Prepared base alpha data in {time.perf_counter() - prep_start:.2f}s")

    diagnostics = []
    alpha_outputs: dict[str, pd.DataFrame] = {}
    baseline_key = make_alpha_scenario_key(baseline_horizon_minutes, baseline_rho)

    for h in horizons_minutes:
        h_start = time.perf_counter()
        h_config = replace(base_config, horizon_minutes=float(h))
        horizon_df = add_clock_time_future_returns(base_df, h_config)
        moments = estimate_global_alpha_moments(horizon_df)
        for rho in rhos:
            is_baseline = bool(np.isclose(h, baseline_horizon_minutes) and np.isclose(rho, baseline_rho))
            keep_full = bool(save_all_alpha_outputs or is_baseline)
            scenario_config = replace(
                h_config,
                target_corr=float(rho),
                random_seed=_scenario_seed(random_seed, float(h), float(rho)),
            )
            scenario_key = make_alpha_scenario_key(float(h), float(rho))
            alpha_df, diag = add_alpha_for_rho(
                horizon_df,
                float(rho),
                moments,
                scenario_config.random_seed,
                scenario_config,
                keep_full_output=keep_full,
            )
            diag["recommended_baseline_flag"] = is_baseline
            diagnostics.append(diag)
            _emit_diagnostic_warnings(diag)

            if keep_full:
                if alpha_df is None:
                    raise RuntimeError(f"{scenario_key}: expected full alpha output")
                alpha_df, decay_meta = add_alpha_decay_state_grid_clock_time(
                    alpha_df,
                    decay_half_lives_minutes,
                    alpha_col="alpha_synthetic",
                    config=scenario_config,
                )
                strategy_col = _state_col_name(baseline_decay_half_life_minutes)
                if strategy_col not in alpha_df.columns:
                    raise RuntimeError(f"baseline strategy column missing: {strategy_col}")
                alpha_df["baseline_alpha_for_strategy"] = alpha_df[strategy_col]
                if alpha_df["baseline_alpha_for_strategy"].isna().any():
                    raise RuntimeError("baseline_alpha_for_strategy contains NaNs")
                alpha_outputs[scenario_key] = alpha_df
                if is_baseline:
                    alpha_df[_baseline_columns(alpha_df, scenario_config)].to_csv(
                        output_dir / BASELINE_ALPHA_CLOCK_FILENAME, index=False
                    )
                    alpha_df[_baseline_columns(alpha_df, scenario_config)].to_csv(
                        output_dir / BASELINE_ALPHA_COMPAT_FILENAME, index=False
                    )
                    decay_meta.to_csv(output_dir / "synthetic_alpha_decay_metadata.csv", index=False)
                    decay_diag = alpha_decay_state_diagnostics_clock_time(alpha_df, decay_meta)
                    decay_diag.to_csv(output_dir / "synthetic_alpha_decay_diagnostics.csv", index=False)
                    if (alpha_df["alpha_synthetic"] == 0).all():
                        _warn("baseline alpha_synthetic is all zeros")
                    if make_plots:
                        plot_baseline_distributions_clock_time(alpha_df, fig_dir)
                        plot_alpha_decay_diagnostics(decay_diag, fig_dir)
                        plot_baseline_state_sample_path(alpha_df, strategy_col, fig_dir, scenario_config)
                elif save_all_alpha_outputs:
                    alpha_df.to_csv(output_dir / f"synthetic_alpha_{scenario_key}.csv", index=False)
        print(f"Processed clock-time horizon h={float(h):g}m in {time.perf_counter() - h_start:.2f}s")

    diagnostics_df = pd.DataFrame(diagnostics).sort_values(
        ["horizon_minutes", "target_corr"]
    ).reset_index(drop=True)
    expected = len(horizons_minutes) * len(rhos)
    if len(diagnostics_df) != expected:
        raise RuntimeError(f"expected {expected} diagnostics rows, got {len(diagnostics_df)}")
    if baseline_key not in alpha_outputs:
        raise RuntimeError(f"baseline key {baseline_key} missing from alpha_outputs")
    diagnostics_df.to_csv(output_dir / DIAGNOSTICS_FILENAME, index=False)
    if make_plots:
        plot_empirical_vs_target_corr_clock_time(diagnostics_df, fig_dir)
        plot_heatmap_from_diagnostics_clock_time(
            diagnostics_df,
            "corr_abs_error",
            fig_dir / "corr_abs_error_heatmap_clock_time.png",
            "Absolute Correlation Calibration Error",
        )
        plot_heatmap_from_diagnostics_clock_time(
            diagnostics_df,
            "realized_unbiased_slope",
            fig_dir / "unbiased_slope_heatmap_clock_time.png",
            "Realized Unbiased-Predictor Slope",
        )
        plot_future_return_vol_by_horizon_minutes(diagnostics_df, fig_dir)
    _write_clock_time_summary(output_dir, diagnostics_df, baseline_decay_half_life_minutes)
    print(f"Clock-time synthetic alpha grid completed in {time.perf_counter() - total_start:.2f}s")
    return alpha_outputs, diagnostics_df


def run_synthetic_alpha_grid(
    bin_df: pd.DataFrame,
    horizons: list[int] | None = None,
    rhos: list[float] | None = None,
    baseline_h: int | None = None,
    baseline_rho: float = 0.10,
    output_dir: Path = Path("outputs") / "alphas",
    random_seed: int = 42,
    save_all_alpha_outputs: bool = False,
    make_plots: bool = True,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Backward-compatible wrapper using clock-time minutes.

    Older calls named the horizon argument `horizons`; values are now interpreted
    as minutes, not rows.
    """

    _warn("run_synthetic_alpha_grid now interprets horizons as clock-time minutes, not rows")
    return run_synthetic_alpha_grid_clock_time(
        bin_df=bin_df,
        horizons_minutes=[float(x) for x in (horizons or [1, 5, 10])],
        rhos=rhos or [0.05, 0.10, 0.20, 0.30, 0.50],
        baseline_horizon_minutes=float(baseline_h if baseline_h is not None else 5.0),
        baseline_rho=baseline_rho,
        output_dir=output_dir,
        random_seed=random_seed,
        save_all_alpha_outputs=save_all_alpha_outputs,
        make_plots=make_plots,
    )


def build_synthetic_alpha(df: pd.DataFrame, config: AlphaConfig) -> pd.DataFrame:
    """Compatibility wrapper for one clock-time alpha scenario."""

    return build_synthetic_alpha_clock_time(df, config)


def build_all_alphas(df: pd.DataFrame, config: AlphaConfig) -> pd.DataFrame:
    """Compatibility wrapper returning one clock-time synthetic alpha."""

    return build_synthetic_alpha_clock_time(df, config)


def alpha_diagnostics(alpha_df: pd.DataFrame) -> dict[str, float | int | bool]:
    """Diagnostics for a materialized clock-time alpha DataFrame."""

    valid = alpha_df["valid_future_return"].to_numpy(dtype=bool)
    return _diagnostics_from_valid_arrays(
        alpha_df,
        AlphaConfig(horizon_minutes=float(alpha_df["horizon_minutes"].iloc[0])),
        float(alpha_df["target_corr"].iloc[0]),
        float(alpha_df["alpha_x"].iloc[0]),
        float(alpha_df["alpha_y"].iloc[0]),
        alpha_df.loc[valid, "alpha_synthetic"].to_numpy(dtype=float),
    )


def plot_empirical_vs_target_corr_clock_time(diagnostics_df: pd.DataFrame, fig_dir: Path) -> None:
    """Plot empirical versus target correlation for clock-time horizons."""

    fig, ax = plt.subplots(figsize=(8, 5))
    for horizon, group in diagnostics_df.groupby("horizon_minutes", sort=True):
        group = group.sort_values("target_corr")
        ax.plot(group["target_corr"], group["empirical_corr_alpha_return"], marker="o", label=f"h={horizon:g}m")
    low, high = diagnostics_df["target_corr"].min(), diagnostics_df["target_corr"].max()
    ax.plot([low, high], [low, high], linestyle="--", color="black", label="target")
    ax.set_title("Synthetic Alpha: Empirical vs Target Correlation")
    ax.set_xlabel("Target correlation")
    ax.set_ylabel("Empirical correlation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "empirical_vs_target_corr_clock_time.png", dpi=150)
    plt.close(fig)


def plot_heatmap_from_diagnostics_clock_time(
    diagnostics_df: pd.DataFrame, value_col: str, fig_path: Path, title: str
) -> None:
    """Plot annotated horizon-minutes by rho heatmap."""

    pivot = diagnostics_df.pivot(index="horizon_minutes", columns="target_corr", values=value_col)
    fig, ax = plt.subplots(figsize=(8, 5))
    values = pivot.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("target_corr")
    ax.set_ylabel("horizon_minutes")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{v:g}" for v in pivot.index])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def plot_future_return_vol_by_horizon_minutes(diagnostics_df: pd.DataFrame, fig_dir: Path) -> None:
    """Plot future-return volatility by clock-time horizon."""

    one = diagnostics_df.sort_values(["horizon_minutes", "target_corr"]).drop_duplicates("horizon_minutes")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(one["horizon_minutes"], one["future_return_std"] * 10_000.0, marker="o")
    ax.set_title("Future Return Volatility by Horizon")
    ax.set_xlabel("Horizon minutes")
    ax.set_ylabel("Future return std (bps)")
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "future_return_vol_by_horizon_minutes.png", dpi=150)
    plt.close(fig)


def plot_baseline_distributions_clock_time(alpha_df: pd.DataFrame, fig_dir: Path) -> None:
    """Save baseline raw alpha and future-return distributions."""

    valid = alpha_df.loc[alpha_df["valid_future_return"]]
    for col, path, title in [
        ("alpha_synthetic", "baseline_alpha_distribution_clock_time.png", "Baseline raw alpha"),
        ("future_return_h", "baseline_future_return_distribution_clock_time.png", "Baseline future return"),
    ]:
        series = valid[col].dropna()
        if series.empty:
            continue
        lo, hi = series.quantile(0.005), series.quantile(0.995)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(series.clip(lo, hi), bins=100)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(Path(fig_dir) / path, dpi=150)
        plt.close(fig)


def plot_baseline_state_sample_path(
    alpha_df: pd.DataFrame, state_col: str, fig_dir: Path, config: AlphaConfig
) -> None:
    """Plot a sample raw alpha and decayed state path."""

    group_key = alpha_df.groupby([config.stock_col, config.date_col], sort=False).size().idxmax()
    sample = alpha_df.loc[
        (alpha_df[config.stock_col] == group_key[0]) & (alpha_df[config.date_col] == group_key[1])
    ].head(400)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sample[config.timestamp_col], sample["alpha_synthetic"], linewidth=1, label="raw alpha")
    ax.plot(sample[config.timestamp_col], sample[state_col], linewidth=1, label=state_col)
    ax.set_title("Baseline alpha state sample path")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("alpha")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "baseline_alpha_state_sample_path_clock_time.png", dpi=150)
    plt.close(fig)


def plot_alpha_decay_diagnostics(decay_diag: pd.DataFrame, fig_dir: Path) -> None:
    """Plot decay-state correlation and volatility by half-life."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(decay_diag["half_life_minutes"], decay_diag["corr_alpha_state_future_return"], marker="o")
    ax.set_xscale("log")
    ax.set_title("Alpha state correlation by half-life")
    ax.set_xlabel("Half-life minutes")
    ax.set_ylabel("corr(alpha state, future return)")
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "alpha_decay_corr_by_half_life_minutes.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(decay_diag["half_life_minutes"], decay_diag["alpha_state_std_bps"], marker="o")
    ax.set_xscale("log")
    ax.set_title("Alpha state volatility by half-life")
    ax.set_xlabel("Half-life minutes")
    ax.set_ylabel("state std (bps)")
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "alpha_state_std_by_half_life_minutes.png", dpi=150)
    plt.close(fig)


def _write_clock_time_summary(
    output_dir: Path, diagnostics_df: pd.DataFrame, baseline_decay_half_life_minutes: float
) -> None:
    baseline = diagnostics_df.loc[diagnostics_df["recommended_baseline_flag"].astype(bool)].iloc[0]
    lines = [
        "Synthetic Alpha Validation Summary",
        "==================================",
        "Horizons are clock-time horizons in minutes.",
        "The old row-based h convention is not used as the final convention.",
        "The previous h=5 rows was approximately 50 seconds when bins are 10 seconds.",
        f"Final baseline uses h={baseline['horizon_minutes']:g} minutes, rho={baseline['target_corr']:.2f}, H={baseline_decay_half_life_minutes:g} minutes.",
        "Forecast horizon h controls which future return is predicted.",
        "Decay half-life H controls signal persistence.",
        "Future return lookup uses first available timestamp >= t+h within same stock/date, with tolerance.",
        "Alpha decay uses actual timestamp gaps, not row count.",
        "",
        f"Baseline empirical correlation: {baseline['empirical_corr_alpha_return']:.4f}",
        f"Baseline correlation error: {baseline['corr_abs_error']:.4f}",
        f"Baseline realized unbiased slope: {baseline['realized_unbiased_slope']:.4f}",
    ]
    (Path(output_dir) / "synthetic_alpha_validation_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
