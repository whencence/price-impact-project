"""Scaling factors for OW impact normalization from binSamples."""

from pathlib import Path

import numpy as np
import pandas as pd


def _warn(message: str) -> None:
    print(f"WARNING: {message}")


def _parse_date_series(values: pd.Series) -> pd.Series:
    as_string = values.astype("string").str.strip()
    yyyymmdd = as_string.str.fullmatch(r"\d{8}", na=False)
    parsed_default = pd.to_datetime(as_string.where(~yyyymmdd), errors="coerce")
    parsed_numeric = pd.to_datetime(as_string.where(yyyymmdd), format="%Y%m%d", errors="coerce")
    return parsed_default.fillna(parsed_numeric)


def _parse_time_to_timedelta(values: pd.Series) -> pd.Series:
    as_string = values.astype("string").str.strip()
    has_colon = as_string.str.contains(":", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="timedelta64[ns]")
    parsed.loc[has_colon] = pd.to_timedelta(as_string.loc[has_colon], errors="coerce")
    numeric = pd.to_numeric(values, errors="coerce")
    unresolved = parsed.isna() & numeric.notna()
    if unresolved.any():
        nums = numeric.loc[unresolved]
        numeric_strings = as_string.loc[unresolved].str.replace(r"\.0+$", "", regex=True)
        hhmmss_like = (
            numeric_strings.str.fullmatch(r"\d{5,6}", na=False)
            & nums.gt(24 * 60 * 60)
            & nums.between(0, 235959)
        )
        if hhmmss_like.any():
            padded = numeric_strings.loc[hhmmss_like].str.zfill(6)
            hh = pd.to_numeric(padded.str.slice(0, 2), errors="coerce")
            mm = pd.to_numeric(padded.str.slice(2, 4), errors="coerce")
            ss = pd.to_numeric(padded.str.slice(4, 6), errors="coerce")
            parsed.loc[padded.index] = (
                pd.to_timedelta(hh, unit="h")
                + pd.to_timedelta(mm, unit="m")
                + pd.to_timedelta(ss, unit="s")
            )
        remaining = unresolved & parsed.isna() & numeric.notna()
        nums = numeric.loc[remaining]
        if len(nums):
            seconds_mask = nums.between(0, 24 * 60 * 60)
            parsed.loc[seconds_mask.index[seconds_mask]] = pd.to_timedelta(nums.loc[seconds_mask], unit="s")
        remaining = unresolved & parsed.isna() & numeric.notna()
        nums = numeric.loc[remaining]
        if len(nums):
            ms_mask = nums.between(0, 24 * 60 * 60 * 1_000)
            parsed.loc[ms_mask.index[ms_mask]] = pd.to_timedelta(nums.loc[ms_mask], unit="ms")
        remaining = unresolved & parsed.isna() & numeric.notna()
        nums = numeric.loc[remaining]
        if len(nums):
            padded = nums.astype("Int64").astype("string").str.zfill(6)
            hh = pd.to_numeric(padded.str.slice(0, 2), errors="coerce")
            mm = pd.to_numeric(padded.str.slice(2, 4), errors="coerce")
            ss = pd.to_numeric(padded.str.slice(4, 6), errors="coerce")
            parsed.loc[remaining] = (
                pd.to_timedelta(hh, unit="h")
                + pd.to_timedelta(mm, unit="m")
                + pd.to_timedelta(ss, unit="s")
            )
    return parsed


def parse_bin_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Parse binSamples date/time into date_parsed and timestamp columns."""

    required = {"date", "time", "stock"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"binSamples missing required timestamp columns: {sorted(missing)}")
    out = df.copy()
    out["date_parsed"] = _parse_date_series(out["date"])
    out["timestamp"] = out["date_parsed"] + _parse_time_to_timedelta(out["time"])
    if out["timestamp"].isna().any():
        _warn(f"{int(out['timestamp'].isna().sum())} rows have unparseable timestamps")
    return out.sort_values(["stock", "date_parsed", "timestamp"]).reset_index(drop=True)


def compute_daily_stock_info_from_bins(bin_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily stock volatility and traded volume from binSamples."""

    if "trade" not in bin_df.columns:
        raise ValueError("binSamples must contain trade to compute daily volume")
    price_col = "midEnd" if "midEnd" in bin_df.columns else "mid"
    if price_col == "mid":
        _warn("midEnd missing; falling back to mid for daily volatility")
    if price_col not in bin_df.columns:
        raise ValueError("binSamples must contain midEnd or mid")

    data = parse_bin_timestamp(bin_df)
    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")
    data["trade"] = pd.to_numeric(data["trade"], errors="coerce").fillna(0.0)
    data[price_col] = data.groupby(["stock", "date_parsed"], sort=False)[price_col].ffill().bfill()
    data["ret"] = data.groupby(["stock", "date_parsed"], sort=False)[price_col].pct_change()

    daily = (
        data.groupby(["stock", "date_parsed"], sort=False)
        .agg(
            px_vol=("ret", lambda x: float(x.std(ddof=1))),
            daily_volume=("trade", lambda x: float(np.abs(x).sum())),
            n_obs=("stock", "size"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
        )
        .reset_index()
        .rename(columns={"date_parsed": "date"})
    )
    return daily


def load_all_bin_samples(
    bin_dir: Path, max_files: int | None = None, nrows: int | None = None
) -> pd.DataFrame:
    """Load and concatenate all binSample CSV files under bin_dir."""

    bin_dir = Path(bin_dir)
    files = sorted(bin_dir.rglob("*.csv"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"no binSample CSV files found under {bin_dir}")
    frames = []
    usecols = [
        "date",
        "time",
        "stock",
        "trade",
        "mid",
        "midEnd",
    ]
    for path in files:
        frame = pd.read_csv(path, usecols=lambda col: col in usecols, nrows=nrows, low_memory=False)
        frame["source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def compute_trailing_scaling_factors(
    daily_info_df: pd.DataFrame, window_days: int = 20
) -> pd.DataFrame:
    """Compute previous-window trailing sigma and ADV by stock, excluding current day."""

    required = {"stock", "date", "px_vol", "daily_volume"}
    missing = required.difference(daily_info_df.columns)
    if missing:
        raise ValueError(f"daily_info_df missing required columns: {sorted(missing)}")
    out = daily_info_df.copy().sort_values(["stock", "date"]).reset_index(drop=True)
    out["trailing_px_vol"] = (
        out.groupby("stock", sort=False)["px_vol"]
        .transform(lambda s: s.rolling(window_days, min_periods=window_days).mean().shift(1))
    )
    out["trailing_ADV"] = (
        out.groupby("stock", sort=False)["daily_volume"]
        .transform(lambda s: s.rolling(window_days, min_periods=window_days).mean().shift(1))
    )
    out["scaling_window_days"] = (
        out.groupby("stock", sort=False).cumcount().clip(upper=window_days)
    )
    out["has_full_scaling_window"] = out["scaling_window_days"] >= window_days
    return out[
        [
            "stock",
            "date",
            "trailing_px_vol",
            "trailing_ADV",
            "px_vol",
            "daily_volume",
            "scaling_window_days",
            "has_full_scaling_window",
        ]
    ].rename(
        columns={
            "px_vol": "px_vol_current_day",
            "daily_volume": "daily_volume_current_day",
        }
    )


def compute_or_load_scaling_factors(
    bin_dir: Path,
    output_path: Path,
    force_recompute: bool = False,
    window_days: int = 20,
    max_files: int | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load existing scaling factors or compute and save them from binSamples."""

    output_path = Path(output_path)
    if output_path.exists() and not force_recompute:
        return pd.read_csv(output_path, parse_dates=["date"])
    files = sorted(Path(bin_dir).rglob("*.csv"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"no binSample CSV files found under {bin_dir}")
    daily_frames = []
    usecols = ["date", "time", "stock", "trade", "mid", "midEnd"]
    for path in files:
        frame = pd.read_csv(path, usecols=lambda col: col in usecols, nrows=nrows, low_memory=False)
        daily_frames.append(compute_daily_stock_info_from_bins(frame))
    daily = pd.concat(daily_frames, ignore_index=True)
    daily = (
        daily.groupby(["stock", "date"], as_index=False, sort=False)
        .agg(
            px_vol=("px_vol", "mean"),
            daily_volume=("daily_volume", "sum"),
            n_obs=("n_obs", "sum"),
            first_timestamp=("first_timestamp", "min"),
            last_timestamp=("last_timestamp", "max"),
        )
    )
    scaling = compute_trailing_scaling_factors(daily, window_days=window_days)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scaling.to_csv(output_path, index=False)
    return scaling
