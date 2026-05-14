"""Load and standardize teammate rolling data for integrated simulations."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.integration_config import IntegratedRunConfig, TeammateDataConfig


BASELINE_20_FALLBACK = [
    "AAL", "AAP", "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADP", "ADSK",
    "AGN", "ALGN", "ALXN", "AMAT", "AMD", "AMGN", "AMT", "AMZN", "ANTM", "APD",
]


def resolve_project_root(start: Path | None = None) -> Path:
    """Resolve repository root."""

    start = (start or Path.cwd()).resolve()
    if start.name in {"src", "notebooks"}:
        start = start.parent
    for candidate in [start] + list(start.parents):
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not find project root")


def _resolve_teammate_path(value: str, project_root: Path, config: TeammateDataConfig) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    name = path.name
    if "bin_" in name:
        return project_root / config.monthly_bin_dir / name
    if "fills_" in name or "fill_" in name:
        return project_root / config.monthly_fill_dir / name
    if not path.is_absolute():
        return project_root / path
    return project_root / str(path).split("processed_2_1/")[-1] if "processed_2_1/" in str(path) else path


def load_rolling_pair_summary(config: TeammateDataConfig, project_root: Path | None = None) -> pd.DataFrame:
    """Load rolling pair summary and resolve train/test paths."""

    root = project_root or resolve_project_root()
    path = root / config.rolling_pair_summary_path
    df = pd.read_csv(path)
    required = {
        "pair_id", "train_month", "test_month", "train_bin_path", "test_bin_path",
        "train_fill_path", "test_fill_path", "n_train_stocks", "n_test_stocks",
        "n_common_stocks_pair_universe", "top20_common_stocks_by_train_notional",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"rolling_pair_summary missing columns: {sorted(missing)}")
    df = df.copy()
    df["pair_id"] = df["pair_id"].astype(int)
    df["train_month"] = df["train_month"].astype(str)
    df["test_month"] = df["test_month"].astype(str)
    for col in ["train_bin_path", "test_bin_path", "train_fill_path", "test_fill_path"]:
        df[col] = df[col].map(lambda x: str(_resolve_teammate_path(x, root, config)))
    return df


def load_baseline_20stocks(config: TeammateDataConfig, project_root: Path | None = None) -> list[str]:
    """Load baseline 20-stock universe, falling back to audited list if needed."""

    root = project_root or resolve_project_root()
    try:
        df = pd.read_csv(root / config.baseline_20stocks_summary_path)
        if "stock" in df.columns:
            stocks = sorted(df["stock"].dropna().astype(str).unique())
            if stocks:
                return stocks
    except Exception as exc:
        print(f"WARNING: failed to parse baseline 20 stocks; using fallback list. {exc}")
    return BASELINE_20_FALLBACK


def parse_pair_universe(pair_row: pd.Series) -> list[str]:
    """Parse top20_common_stocks_by_train_notional from rolling pair row."""

    value = pair_row.get("top20_common_stocks_by_train_notional", "")
    if isinstance(value, list):
        return [str(x) for x in value]
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [item.strip().strip("'\"") for item in text.split(",") if item.strip()]


def _read_table(path: Path, columns: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(path, columns=columns)
        except ImportError as exc:
            raise ImportError(
                f"Cannot read parquet file {path}. Install pyarrow or fastparquet in this environment, "
                "or ask teammate to export the processed parquet files to CSV for integration."
            ) from exc
        return df.head(nrows) if nrows is not None else df
    return pd.read_csv(path, usecols=lambda c: columns is None or c in columns, nrows=nrows, low_memory=False)


def standardize_teammate_bin_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize teammate bin data to columns used by my modules."""

    out = df.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    elif "timestamp" in out.columns:
        out["datetime"] = pd.to_datetime(out["timestamp"], errors="coerce")
    else:
        date = pd.to_datetime(out["date"].astype(str), format="%Y%m%d", errors="coerce")
        date = date.fillna(pd.to_datetime(out["date"], errors="coerce"))
        out["datetime"] = date + pd.to_timedelta(out["time"].astype(str), errors="coerce")
    out["timestamp"] = out["datetime"]
    if "trading_date" not in out.columns:
        out["trading_date"] = out["datetime"].dt.date.astype(str)
    else:
        out["trading_date"] = pd.to_datetime(out["trading_date"], errors="coerce").dt.date.astype(str)
    out["date"] = out["trading_date"]
    if "seconds_from_open" not in out.columns:
        open_ts = pd.to_datetime(out["trading_date"]) + pd.Timedelta(hours=9, minutes=30)
        out["seconds_from_open"] = (out["datetime"] - open_ts).dt.total_seconds()
    for col in ["mid", "midEnd", "orderFlow", "trade", "hidden", "auction", "spread", "depth", "lobImb", "effLobImb"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = 0.0 if col not in {"mid", "midEnd"} else pd.NA
    return out.sort_values(["stock", "trading_date", "datetime"]).reset_index(drop=True)


def _balanced_stock_head(df: pd.DataFrame, stocks: list[str], nrows: int | None) -> pd.DataFrame:
    """Return a debug-sized sample spread across selected stocks."""

    if nrows is None or len(df) <= nrows:
        return df.copy()
    n_stocks = max(len(stocks), 1)
    per_stock = max(int(np.ceil(nrows / n_stocks)), 1)
    sampled = (
        df.sort_values(["stock", "datetime"] if "datetime" in df.columns else ["stock"])
        .groupby("stock", sort=False, observed=True)
        .head(per_stock)
        .head(nrows)
        .copy()
    )
    return sampled


def load_monthly_bin_data_for_pair(
    pair_row: pd.Series,
    stocks: list[str],
    config: TeammateDataConfig,
    sample_nrows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test bin data for a rolling pair and stock universe."""

    train_path = Path(str(pair_row["train_bin_path"]))
    test_path = Path(str(pair_row["test_bin_path"]))
    columns = [
        "date", "time", "stock", "trade", "orderFlow", "hidden", "auction", "mid", "midEnd",
        "spread", "depth", "lobImb", "effLobImb", "datetime", "trading_date", "seconds_from_open",
    ]
    # Read first, then apply the stock filter and optional debug row cap. For
    # parquet we read the full selected-column table anyway, so this makes
    # --max-rows-per-pair mean rows in the selected universe rather than the
    # first rows of the raw file, which can accidentally keep only one stock.
    train = _read_table(train_path, columns=columns)
    test = _read_table(test_path, columns=columns)
    train = train[train["stock"].astype(str).isin(stocks)].copy()
    test = test[test["stock"].astype(str).isin(stocks)].copy()
    train = _balanced_stock_head(train, stocks, sample_nrows)
    test = _balanced_stock_head(test, stocks, sample_nrows)
    return standardize_teammate_bin_df(train), standardize_teammate_bin_df(test)


def get_pair_records(
    data_config: TeammateDataConfig,
    run_config: IntegratedRunConfig,
    project_root: Path | None = None,
) -> list[pd.Series]:
    """Return rolling pair rows selected by run configuration."""

    pairs = load_rolling_pair_summary(data_config, project_root)
    if run_config.selected_pair_ids is not None:
        pairs = pairs[pairs["pair_id"].isin(run_config.selected_pair_ids)]
    elif run_config.run_mode == "single_pair" and run_config.debug_pair_id is not None:
        pairs = pairs[pairs["pair_id"].eq(run_config.debug_pair_id)]
    if run_config.max_pairs is not None:
        pairs = pairs.head(run_config.max_pairs)
    return [row for _, row in pairs.iterrows()]
