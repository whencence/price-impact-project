"""Utility functions for the data audit notebook.

The notebook keeps the analysis flow visible, while this module holds repeated
data-cleaning, validation, plotting, and summary helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


BIN_REQUIRED_COLUMNS = [
    "date",
    "time",
    "stock",
    "trade",
    "orderFlow",
    "mid",
    "midEnd",
    "spread",
    "effSpread",
    "depth",
    "nbEvents",
    "nbTrades",
]

FILL_REQUIRED_COLUMNS = [
    "date",
    "stock",
    "time",
    "trade",
    "mid",
    "spread",
    "effSpread",
    "depth",
    "lobImb",
    "ask",
    "bid",
    "askVolume",
    "bidVolume",
]


def file_inventory(files: Iterable[Path]) -> pd.DataFrame:
    """Return a file inventory table with paths, names, and sizes."""

    rows = []
    for path in files:
        rows.append(
            {
                "file": str(path),
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "size_mb": path.stat().st_size / 1024**2,
            }
        )
    return pd.DataFrame(rows)


def load_csv_files(
    files: list[Path],
    sample_type: str,
    max_files: int | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load and concatenate CSV files, tagging source_file and sample_type."""

    selected_files = files[:max_files] if max_files is not None else files
    frames = []
    for path in selected_files:
        frame = pd.read_csv(path, nrows=nrows, low_memory=False)
        frame["source_file"] = path.name
        frame["sample_type"] = sample_type
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_date_series(values: pd.Series) -> pd.Series:
    """Parse common course-date encodings into pandas datetime dates."""

    as_string = values.astype("string").str.strip()
    numeric_like = as_string.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.to_datetime(as_string.where(~numeric_like), errors="coerce")
    parsed_numeric = pd.to_datetime(as_string.where(numeric_like), format="%Y%m%d", errors="coerce")
    return parsed.fillna(parsed_numeric)


def parse_time_to_timedelta(values: pd.Series) -> pd.Series:
    """Parse common intraday time encodings into timedeltas from midnight."""

    as_string = values.astype("string").str.strip()
    timedeltas = pd.to_timedelta(as_string, errors="coerce")
    numeric = pd.to_numeric(values, errors="coerce")
    unresolved = timedeltas.isna() & numeric.notna()
    if unresolved.any():
        nums = numeric[unresolved]
        max_value = nums.abs().max()
        if max_value <= 24 * 60 * 60:
            timedeltas.loc[unresolved] = pd.to_timedelta(nums, unit="s")
        elif max_value <= 24 * 60 * 60 * 1000:
            timedeltas.loc[unresolved] = pd.to_timedelta(nums, unit="ms")
        else:
            # Handles HHMMSS or HHMMSSmmm style numeric encodings.
            padded = nums.astype("Int64").astype("string").str.zfill(6)
            hh = pd.to_numeric(padded.str.slice(0, 2), errors="coerce")
            mm = pd.to_numeric(padded.str.slice(2, 4), errors="coerce")
            ss = pd.to_numeric(padded.str.slice(4, 6), errors="coerce")
            timedeltas.loc[unresolved] = (
                pd.to_timedelta(hh, unit="h")
                + pd.to_timedelta(mm, unit="m")
                + pd.to_timedelta(ss, unit="s")
            )
    return timedeltas


def standardize_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted copy with date_parsed and timestamp columns."""

    if df.empty:
        return df.copy()
    required = {"date", "time", "stock"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"cannot parse timestamps; missing columns: {sorted(missing)}")
    out = df.copy()
    out["date_parsed"] = parse_date_series(out["date"])
    time_delta = parse_time_to_timedelta(out["time"])
    out["timestamp"] = out["date_parsed"] + time_delta
    return out.sort_values(["stock", "date_parsed", "timestamp"]).reset_index(drop=True)


def stock_date_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count rows per stock-date and summarize timestamp coverage."""

    if df.empty or not {"stock", "date_parsed"}.issubset(df.columns):
        return pd.DataFrame()
    grouped = df.groupby(["stock", "date_parsed"], dropna=False, sort=False)
    summary = grouped.agg(
        rows=("stock", "size"),
        first_timestamp=("timestamp", "min") if "timestamp" in df.columns else ("stock", "size"),
        last_timestamp=("timestamp", "max") if "timestamp" in df.columns else ("stock", "size"),
    )
    if "timestamp" in df.columns:
        summary["n_timestamps"] = grouped["timestamp"].nunique(dropna=True)
    return summary.reset_index()


def schema_validation_report(
    df: pd.DataFrame,
    required_columns: list[str],
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build a schema and missingness validation report for one dataset."""

    expected = set(expected_columns or required_columns)
    observed = set(df.columns)
    rows = []
    all_columns = sorted(observed.union(expected))
    key_cols = [col for col in ["stock", "date", "time"] if col in df.columns]
    duplicate_key_rows = int(df.duplicated(key_cols).sum()) if key_cols else 0
    duplicate_rows = int(df.duplicated().sum()) if not df.empty else 0
    for col in all_columns:
        rows.append(
            {
                "column": col,
                "present": col in observed,
                "required": col in required_columns,
                "missing_column": col in required_columns and col not in observed,
                "extra_column": col in observed and col not in expected,
                "dtype": str(df[col].dtype) if col in observed else "",
                "missing_pct": float(df[col].isna().mean()) if col in observed and len(df) else np.nan,
                "duplicated_rows": duplicate_rows,
                "duplicated_stock_date_time_rows": duplicate_key_rows,
            }
        )
    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize numeric columns with quantiles, signs, zeros, and missingness."""

    if df.empty:
        return pd.DataFrame()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for col in numeric_cols:
        series = df[col]
        clean = series.dropna()
        rows.append(
            {
                "column": col,
                "count": int(clean.count()),
                "mean": float(clean.mean()) if len(clean) else np.nan,
                "std": float(clean.std(ddof=1)) if len(clean) > 1 else np.nan,
                "min": float(clean.min()) if len(clean) else np.nan,
                "q01": float(clean.quantile(0.01)) if len(clean) else np.nan,
                "q05": float(clean.quantile(0.05)) if len(clean) else np.nan,
                "median": float(clean.median()) if len(clean) else np.nan,
                "q95": float(clean.quantile(0.95)) if len(clean) else np.nan,
                "q99": float(clean.quantile(0.99)) if len(clean) else np.nan,
                "max": float(clean.max()) if len(clean) else np.nan,
                "missing_pct": float(series.isna().mean()),
                "n_zeros": int((clean == 0).sum()),
                "n_negative": int((clean < 0).sum()),
                "n_positive": int((clean > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def data_quality_checks(df: pd.DataFrame, sample_type: str) -> pd.DataFrame:
    """Run core sanity checks for prices, spreads, depths, counts, and imbalance."""

    if df.empty:
        return pd.DataFrame()
    checks: list[dict[str, object]] = []

    def add_check(name: str, mask: pd.Series, description: str) -> None:
        valid_mask = mask.fillna(False)
        checks.append(
            {
                "sample_type": sample_type,
                "check": name,
                "description": description,
                "n_fail": int((~valid_mask).sum()),
                "fail_pct": float((~valid_mask).mean()),
                "n_missing_in_check": int(mask.isna().sum()),
            }
        )

    if "mid" in df.columns:
        add_check("mid_positive", df["mid"] > 0, "mid should be strictly positive")
    for col in ["spread", "effSpread", "depth", "nbEvents", "nbTrades", "nbHidden", "askVolume", "bidVolume"]:
        if col in df.columns:
            add_check(f"{col}_non_negative", df[col] >= 0, f"{col} should usually be non-negative")
    if {"ask", "bid"}.issubset(df.columns):
        add_check("ask_ge_bid", df["ask"] >= df["bid"], "ask should be at least bid")
    if {"mid", "ask", "bid"}.issubset(df.columns):
        tolerance = np.maximum(1e-8, 0.01 * (df["ask"] - df["bid"]).abs())
        add_check(
            "mid_between_bid_ask",
            (df["mid"] >= df["bid"] - tolerance) & (df["mid"] <= df["ask"] + tolerance),
            "mid should sit inside or very near bid-ask bounds",
        )
    if {"spread", "ask", "bid"}.issubset(df.columns):
        diff = (df["spread"] - (df["ask"] - df["bid"])).abs()
        add_check(
            "spread_matches_ask_minus_bid",
            diff <= np.maximum(1e-8, 0.05 * df["spread"].abs()),
            "spread should approximately equal ask - bid when units match",
        )
    for col in ["lobImb", "effLobImb"]:
        if col in df.columns:
            add_check(f"{col}_mostly_unit_interval", df[col].between(-1.05, 1.05), f"{col} should usually be in [-1, 1]")
    return pd.DataFrame(checks)


def save_missing_plot(df: pd.DataFrame, title: str, path: Path) -> pd.DataFrame:
    """Save a bar chart of missing percentages and return the plotted table."""

    missing = df.isna().mean().sort_values(ascending=False).rename("missing_pct").reset_index()
    missing = missing.rename(columns={"index": "column"})
    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(missing))))
    ax.barh(missing["column"], missing["missing_pct"])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Missing fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return missing


def add_future_returns(df: pd.DataFrame, horizons: list[int], price_col: str = "mid") -> pd.DataFrame:
    """Add future return columns by stock-date for the requested horizons."""

    if df.empty:
        return df.copy()
    out = df.copy().sort_values(["stock", "date_parsed", "timestamp"]).reset_index(drop=True)
    grouped = out.groupby(["stock", "date_parsed"], sort=False)
    for h in horizons:
        future_price = grouped[price_col].shift(-h)
        out[f"future_return_h{h}"] = future_price / out[price_col] - 1.0
    return out


def monthly_availability(bin_df: pd.DataFrame, fill_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize month-level trading-day, stock, stock-day, and row coverage."""

    rows = []
    for name, df in [("bin", bin_df), ("fill", fill_df)]:
        if df.empty or "date_parsed" not in df.columns:
            continue
        work = df.copy()
        work["month"] = work["date_parsed"].dt.to_period("M").astype("string")
        grouped = work.groupby("month", sort=True)
        summary = grouped.agg(
            n_rows=("stock", "size"),
            n_dates=("date_parsed", "nunique"),
            n_stocks=("stock", "nunique"),
        ).reset_index()
        stock_days = work.drop_duplicates(["month", "stock", "date_parsed"]).groupby("month").size()
        summary["n_stock_days"] = summary["month"].map(stock_days).astype(int)
        summary["sample_type"] = name
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def candidate_universe(bin_df: pd.DataFrame, fill_df: pd.DataFrame, n_stocks: int = 20) -> pd.DataFrame:
    """Rank stocks by coverage and basic quality for baseline universe selection."""

    stocks = sorted(set(bin_df.get("stock", pd.Series(dtype=object))).union(fill_df.get("stock", pd.Series(dtype=object))))
    rows = []
    for stock in stocks:
        bin_stock = bin_df.loc[bin_df["stock"] == stock].copy() if not bin_df.empty else pd.DataFrame()
        fill_stock = fill_df.loc[fill_df["stock"] == stock].copy() if not fill_df.empty else pd.DataFrame()
        combined_dates = set(bin_stock.get("date_parsed", pd.Series(dtype=object)).dropna()).union(
            fill_stock.get("date_parsed", pd.Series(dtype=object)).dropna()
        )
        months = set(pd.Series(list(combined_dates)).dt.to_period("M").astype("string")) if combined_dates else set()
        daily_rows = bin_stock.groupby("date_parsed").size() if not bin_stock.empty else pd.Series(dtype=float)
        rows.append(
            {
                "stock": stock,
                "number_of_dates": len(combined_dates),
                "number_of_months": len(months),
                "total_rows_binSamples": int(len(bin_stock)),
                "total_rows_fillSamples": int(len(fill_stock)),
                "median_rows_per_day": float(daily_rows.median()) if len(daily_rows) else 0.0,
                "missing_mid_pct": float(bin_stock["mid"].isna().mean()) if "mid" in bin_stock else np.nan,
                "positive_mid_pct": float((bin_stock["mid"] > 0).mean()) if "mid" in bin_stock and len(bin_stock) else np.nan,
                "median_spread": float(bin_stock["spread"].median()) if "spread" in bin_stock and len(bin_stock) else np.nan,
                "median_depth": float(bin_stock["depth"].median()) if "depth" in bin_stock and len(bin_stock) else np.nan,
                "total_nbTrades": float(bin_stock["nbTrades"].sum()) if "nbTrades" in bin_stock else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["quality_score"] = (
        result["number_of_dates"].fillna(0)
        + result["number_of_months"].fillna(0) * 20
        + np.log1p(result["total_rows_binSamples"].fillna(0))
        + np.log1p(result["total_rows_fillSamples"].fillna(0))
    )
    result = result.sort_values(
        ["positive_mid_pct", "number_of_months", "number_of_dates", "median_rows_per_day", "quality_score"],
        ascending=[False, False, False, False, False],
    )
    return result.head(n_stocks).reset_index(drop=True)


def plot_histogram(series: pd.Series, title: str, xlabel: str, path: Path, bins: int = 100, clip_quantile: float = 0.995) -> None:
    """Save a clipped histogram for a numeric series."""

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return
    lower = clean.quantile(1 - clip_quantile)
    upper = clean.quantile(clip_quantile)
    clean = clean.clip(lower, upper)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(clean, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sample_paths(
    df: pd.DataFrame,
    path: Path,
    selected_stocks: list[str] | None = None,
    sample_n_stocks: int = 3,
    sample_n_days: int = 2,
    price_col: str = "mid",
) -> None:
    """Save sample intraday price paths for a few stock-date groups."""

    if df.empty or price_col not in df.columns:
        return
    stocks = selected_stocks or sorted(df["stock"].dropna().unique())[:sample_n_stocks]
    subset = df.loc[df["stock"].isin(stocks)].copy()
    dates = sorted(subset["date_parsed"].dropna().unique())[:sample_n_days]
    subset = subset.loc[subset["date_parsed"].isin(dates)].copy()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for (stock, date_value), group in subset.groupby(["stock", "date_parsed"], sort=False):
        ax.plot(group["timestamp"], group[price_col], linewidth=1, label=f"{stock} {pd.Timestamp(date_value).date()}")
    ax.set_title("Sample mid-price paths")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(price_col)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def large_price_jumps(df: pd.DataFrame, thresholds: tuple[float, ...] = (0.01, 0.05, 0.10)) -> pd.DataFrame:
    """Return rows with large intraday mid-price changes."""

    if df.empty or "mid" not in df.columns:
        return pd.DataFrame()
    work = df.copy().sort_values(["stock", "date_parsed", "timestamp"]).reset_index(drop=True)
    work["ret_mid_1"] = work.groupby(["stock", "date_parsed"], sort=False)["mid"].pct_change()
    rows = []
    for threshold in thresholds:
        hits = work.loc[work["ret_mid_1"].abs() > threshold].copy()
        if hits.empty:
            continue
        hits["threshold"] = threshold
        rows.append(hits)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def correlation_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute a numeric correlation table for available columns."""

    available = [col for col in columns if col in df.columns]
    if len(available) < 2:
        return pd.DataFrame()
    return df[available].corr(numeric_only=True)


def merge_diagnostics(bin_df: pd.DataFrame, fill_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge bin and fill samples on stock/date/time keys and summarize matches."""

    if bin_df.empty or fill_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    key = ["stock", "date_parsed", "time"] if "date_parsed" in bin_df and "date_parsed" in fill_df else ["stock", "date", "time"]
    bin_dups = int(bin_df.duplicated(key).sum())
    fill_dups = int(fill_df.duplicated(key).sum())
    inner = bin_df.merge(fill_df, on=key, how="inner", suffixes=("_bin", "_fill"))
    left_bin = bin_df.merge(fill_df[key].drop_duplicates(), on=key, how="left", indicator=True)
    left_fill = fill_df.merge(bin_df[key].drop_duplicates(), on=key, how="left", indicator=True)
    report = pd.DataFrame(
        [
            {"metric": "bin_rows", "value": len(bin_df)},
            {"metric": "fill_rows", "value": len(fill_df)},
            {"metric": "inner_merge_rows", "value": len(inner)},
            {"metric": "bin_rows_matched_pct", "value": float((left_bin["_merge"] == "both").mean()) if len(left_bin) else 0.0},
            {"metric": "fill_rows_matched_pct", "value": float((left_fill["_merge"] == "both").mean()) if len(left_fill) else 0.0},
            {"metric": "bin_duplicate_keys", "value": bin_dups},
            {"metric": "fill_duplicate_keys", "value": fill_dups},
            {"metric": "many_to_many_merge_risk", "value": bool(bin_dups > 0 and fill_dups > 0)},
        ]
    )
    comparisons = compare_common_columns(inner)
    return inner, report, comparisons


def compare_common_columns(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Compare common numeric columns after bin-fill merge."""

    rows = []
    for col in ["mid", "spread", "effSpread", "depth", "lobImb", "trade"]:
        bin_col = f"{col}_bin"
        fill_col = f"{col}_fill"
        if not {bin_col, fill_col}.issubset(merged_df.columns):
            continue
        left = pd.to_numeric(merged_df[bin_col], errors="coerce")
        right = pd.to_numeric(merged_df[fill_col], errors="coerce")
        diff = left - right
        rows.append(
            {
                "column": col,
                "n_obs": int(diff.notna().sum()),
                "mean_diff": float(diff.mean()),
                "median_diff": float(diff.median()),
                "rmse": float(np.sqrt((diff**2).mean())),
                "correlation": float(left.corr(right)) if diff.notna().sum() > 1 else np.nan,
                "max_abs_diff": float(diff.abs().max()),
            }
        )
    return pd.DataFrame(rows)


def scatter_or_hexbin(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    path: Path,
    sample_size: int = 200_000,
    use_hexbin: bool = True,
) -> None:
    """Save a sampled scatter or hexbin plot for two numeric columns."""

    if df.empty or not {x, y}.issubset(df.columns):
        return
    clean = df[[x, y]].dropna().copy()
    if clean.empty:
        return
    if len(clean) > sample_size:
        clean = clean.sample(sample_size, random_state=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    if use_hexbin:
        hb = ax.hexbin(clean[x], clean[y], gridsize=50, mincnt=1)
        fig.colorbar(hb, ax=ax, label="Count")
    else:
        ax.scatter(clean[x], clean[y], s=4, alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def binned_average(df: pd.DataFrame, x_col: str, y_col: str, n_bins: int = 10) -> pd.DataFrame:
    """Return average y by quantile bins of x."""

    if df.empty or not {x_col, y_col}.issubset(df.columns):
        return pd.DataFrame()
    clean = df[[x_col, y_col]].dropna().copy()
    if clean.empty:
        return pd.DataFrame()
    clean["bin"] = pd.qcut(clean[x_col], q=min(n_bins, clean[x_col].nunique()), duplicates="drop")
    return clean.groupby("bin", observed=True).agg(
        x_mean=(x_col, "mean"),
        y_mean=(y_col, "mean"),
        n_obs=(y_col, "size"),
    ).reset_index(drop=True)


def synthetic_alpha_feasibility(
    df: pd.DataFrame,
    horizons: list[int],
    rhos: list[float],
    random_seed: int = 42,
) -> pd.DataFrame:
    """Check feasibility and empirical correlations for homework synthetic alpha."""

    if df.empty or "mid" not in df.columns:
        return pd.DataFrame()
    rng = np.random.default_rng(random_seed)
    rows = []
    base = df.copy().sort_values(["stock", "date_parsed", "timestamp"]).reset_index(drop=True)
    grouped = base.groupby(["stock", "date_parsed"], sort=False)
    for h in horizons:
        future_return = grouped["mid"].shift(-h) / base["mid"] - 1.0
        valid = future_return.notna() & (base["mid"] > 0)
        r = future_return.loc[valid]
        prices = base.loc[valid, "mid"]
        var_r = float(r.var(ddof=1)) if len(r) > 1 else 0.0
        mean_inv_price_sq = float((1.0 / prices**2).mean()) if len(prices) else np.nan
        for rho in rhos:
            x = rho**2
            if var_r > 0 and mean_inv_price_sq > 0:
                y = rho * np.sqrt(1 - rho**2) * np.sqrt(var_r / (mean_inv_price_sq * h))
                delta_w = np.sqrt(h) * rng.normal(0.0, 1.0, size=len(r))
                alpha = x * r.to_numpy() + y * delta_w / prices.to_numpy()
                corr = float(pd.Series(alpha).corr(pd.Series(r.to_numpy()))) if len(r) > 1 else np.nan
                alpha_mean = float(np.mean(alpha))
                alpha_std = float(np.std(alpha, ddof=1)) if len(alpha) > 1 else 0.0
            else:
                y = 0.0
                corr = np.nan
                alpha_mean = 0.0
                alpha_std = 0.0
            rows.append(
                {
                    "horizon": h,
                    "rho_target": rho,
                    "x": x,
                    "y": y,
                    "var_future_return": var_r,
                    "mean_inv_price_sq": mean_inv_price_sq,
                    "n_valid": int(valid.sum()),
                    "missing_rate_horizon_truncation": float((~valid).mean()),
                    "empirical_corr": corr,
                    "alpha_mean": alpha_mean,
                    "alpha_std": alpha_std,
                }
            )
    return pd.DataFrame(rows)


def stress_test_feasibility(df: pd.DataFrame, liquidation_time: str = "12:00") -> pd.DataFrame:
    """Summarize whether each stock-date can support delay and liquidation tests."""

    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    threshold = pd.to_datetime(liquidation_time).time()
    grouped = df.groupby(["stock", "date_parsed"], sort=False)
    rows = []
    for (stock, date_value), group in grouped:
        times = group["timestamp"].dt.time
        rows.append(
            {
                "stock": stock,
                "date": date_value,
                "first_timestamp": group["timestamp"].min(),
                "last_timestamp": group["timestamp"].max(),
                "number_of_rows": len(group),
                "has_timestamp_before_12": bool((times < threshold).any()),
                "has_timestamp_after_12": bool((times >= threshold).any()),
                "feasible_for_forced_liquidation": bool((times < threshold).any() and (times >= threshold).any()),
            }
        )
    return pd.DataFrame(rows)


def rolling_window_plan(bin_df: pd.DataFrame, candidate: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create possible consecutive monthly train-test windows."""

    if bin_df.empty or "date_parsed" not in bin_df.columns:
        return pd.DataFrame()
    work = bin_df.copy()
    work["month"] = work["date_parsed"].dt.to_period("M").astype("string")
    months = sorted(work["month"].dropna().unique())
    candidate_stocks = set(candidate["stock"]) if candidate is not None and not candidate.empty else set()
    rows = []
    for train_month, test_month in zip(months[:-1], months[1:]):
        train = work.loc[work["month"] == train_month]
        test = work.loc[work["month"] == test_month]
        common = set(train["stock"].unique()).intersection(test["stock"].unique())
        candidate_common = common.intersection(candidate_stocks) if candidate_stocks else common
        rows.append(
            {
                "train_month": train_month,
                "test_month": test_month,
                "number_of_common_stocks": len(common),
                "train_stock_days": int(train.drop_duplicates(["stock", "date_parsed"]).shape[0]),
                "test_stock_days": int(test.drop_duplicates(["stock", "date_parsed"]).shape[0]),
                "candidate_20_available_in_both": len(candidate_common),
                "candidate_stocks": ",".join(sorted(candidate_common)[:20]),
            }
        )
    return pd.DataFrame(rows)


def write_final_report(path: Path, lines: list[str]) -> None:
    """Write final audit report text."""

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

