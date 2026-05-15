# Notebook Code Extracts

## notebooks/2_1_full_as_spec_completed.ipynb

### Cell 2: OW
```python
!pip -q install pyarrow

import os
import re
import gc
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from IPython.display import display

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 160)
```

### Cell 4: rolling, universe, 20, stock, impact
```python
# Raw data location in Drive.
DRIVE_BASE = Path("/content/drive/MyDrive/Quantitative Trading and Price Impact")
BIN_DRIVE = DRIVE_BASE / "binSamples"
FILL_DRIVE = DRIVE_BASE / "fillSamples"

# Local working location in google colab.
LOCAL_BASE = Path("/content/project_data")
BIN_LOCAL = LOCAL_BASE / "binSamples"
FILL_LOCAL = LOCAL_BASE / "fillSamples"

PROCESSED_BASE = LOCAL_BASE / "processed_2_1"
MONTHLY_DIR = PROCESSED_BASE / "monthly_standardised"
BASELINE_DIR = PROCESSED_BASE / "baseline_20stocks"
ROLLING_DIR = PROCESSED_BASE / "rolling_full_universe"
REPORT_DIR = PROCESSED_BASE / "report_tables"

for p in [LOCAL_BASE, BIN_LOCAL, FILL_LOCAL, PROCESSED_BASE, MONTHLY_DIR, BASELINE_DIR, ROLLING_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("Drive base:       ", DRIVE_BASE)
print("Local raw base:   ", LOCAL_BASE)
print("Processed outputs:", PROCESSED_BASE)
```

### Cell 5: impact
```python
# Copy from Drive to local Colab disk.
!rsync -ah --progress "/content/drive/MyDrive/Quantitative Trading and Price Impact/binSamples/" "/content/project_data/binSamples/"
!rsync -ah --progress "/content/drive/MyDrive/Quantitative Trading and Price Impact/fillSamples/" "/content/project_data/fillSamples/"
```

### Cell 7: 20, lambda, OW
```python
def month_from_filename(path: Path) -> str:
    """Extract YYYYMM from a filename such as bin201901.csv or fills201901.csv."""
    m = re.search(r"(20\d{4})", path.name)
    if m is None:
        raise ValueError(f"Could not extract month from filename: {path}")
    return m.group(1)


def list_csv_like_files(folder: Path) -> List[Path]:
    """Return csv/csv-like files, ignoring hidden/system files."""
    if not folder.exists():
        return []
    files = []
    for p in folder.iterdir():
        if p.name.startswith("."):
            continue
        if p.is_file() and p.suffix.lower() in [".csv", ".txt", ""]:
            files.append(p)
    return sorted(files, key=lambda x: month_from_filename(x))

bin_files = list_csv_like_files(BIN_LOCAL)
fill_files = list_csv_like_files(FILL_LOCAL)

assert len(bin_files) > 0, "No binSamples files found. Check the Drive/local path."
assert len(fill_files) > 0, "No fillSamples files found. Check the Drive/local path."
```

### Cell 10: stock, missing, volume, OW, OLS
```python
EXPECTED_BIN_COLS = [
    "date", "time", "stock", "trade", "orderFlow", "hidden", "auction",
    "mid", "midEnd", "spread", "effSpread", "lobImb", "effLobImb",
    "trdLiq", "ofLiq", "depth", "nbEvents", "nbHidden", "nbTrades"
]

EXPECTED_FILL_COLS = [
    "date", "stock", "time", "trade", "mid", "spread", "effSpread", "depth",
    "lobImb", "ask", "bid", "askVolume", "bidVolume"
]


def read_columns(path: Path) -> List[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()

schema_rows = []
for p in bin_files:
    cols = read_columns(p)
    schema_rows.append({
        "kind": "bin",
        "month": month_from_filename(p),
        "file_name": p.name,
        "n_cols": len(cols),
        "matches_expected": cols == EXPECTED_BIN_COLS,
        "missing_expected_cols": sorted(set(EXPECTED_BIN_COLS) - set(cols)),
        "extra_cols": sorted(set(cols) - set(EXPECTED_BIN_COLS)),
    })

for p in fill_files:
    cols = read_columns(p)
    schema_rows.append({
        "kind": "fills",
        "month": month_from_filename(p),
        "file_name": p.name,
        "n_cols": len(cols),
        "matches_expected": cols == EXPECTED_FILL_COLS,
        "missing_expected_cols": sorted(set(EXPECTED_FILL_COLS) - set(cols)),
        "extra_cols": sorted(set(cols) - set(EXPECTED_FILL_COLS)),
    })

schema_check = pd.DataFrame(schema_rows).sort_values(["kind", "month"]).reset_index(drop=True)
display(schema_check)
schema_check.to_csv(REPORT_DIR / "schema_check.csv", index=False)

if not schema_check["matches_expected"].all():
    print("WARNING: At least one file has a non-standard schema. Inspect schema_check before continuing.")
```

### Cell 11: OW
```python
# Show small examples from the first bin and fill files.
sample_bin = pd.read_csv(bin_files[0], nrows=5)
sample_fill = pd.read_csv(fill_files[0], nrows=5)

print("Sample bin rows:")
display(sample_bin)
print("Sample fill rows:")
display(sample_fill)
```

### Cell 13: universe, stock, volume, OW, impact, fit
```python
# dictionary of column names as keys and their types as values respectively

BIN_DTYPES = {
    "date": "string",
    "time": "string",
    "stock": "string",
    "trade": "float64",
    "orderFlow": "float64",
    "hidden": "float64",
    "auction": "float64",
    "mid": "float64",
    "midEnd": "float64",
    "spread": "float64",
    "effSpread": "float64",
    "lobImb": "float64",
    "effLobImb": "float64",
    "trdLiq": "float64",
    "ofLiq": "float64",
    "depth": "float64",
    "nbEvents": "float64",
    "nbHidden": "float64",
    "nbTrades": "float64",
}

FILL_DTYPES = {
    "date": "string",
    "time": "string",
    "stock": "string",
    "trade": "float64",
    "mid": "float64",
    "spread": "float64",
    "effSpread": "float64",
    "depth": "float64",
    "lobImb": "float64",
    "ask": "float64",
    "bid": "float64",
    "askVolume": "float64",
    "bidVolume": "float64",
}


def read_raw_month(path: Path, kind: str) -> pd.DataFrame:
    """Read one raw monthly file with consistent dtypes."""
    if kind == "bin":
        return pd.read_csv(path, dtype=BIN_DTYPES)
    elif kind == "fills":
        return pd.read_csv(path, dtype=FILL_DTYPES)
    else:
        raise ValueError("kind must be 'bin' or 'fills'")


def standardise_intraday_data(df: pd.DataFrame, *, kind: str, month: str, source_file: str) -> pd.DataFrame:
    """Standardise one raw monthly dataframe for efficient downstream use."""
    df = df.copy()

    # Basic identifiers.
    df["month"] = month
    df["source_file"] = source_file
    df["stock"] = df["stock"].astype("string").str.strip()

    # Robust datetime parsing. fillSamples has milliseconds; binSamples usually has 10-second bins.
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    df["trading_date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")

    # Seconds since 09:30:00. This is useful for intraday loops and daily resets later.
    session_open = pd.to_timedelta("09:30:00")
    time_of_day = df["datetime"] - df["datetime"].dt.normalize()
    df["seconds_from_open"] = (time_of_day - session_open).dt.total_seconds().astype("float64")

    # Keep only rows with essential fields. Do not over-clean: later sections may need zeros in trade/orderFlow.
    before = len(df)
    df = df[df["datetime"].notna()]
    df = df[df["stock"].notna() & (df["stock"] != "")]
    df = df[df["mid"].notna() & (df["mid"] > 0)]
    after = len(df)

    # Core trading variables.
    df["abs_trade"] = df["trade"].abs()
    df["signed_notional"] = df["trade"] * df["mid"]
    df["abs_notional"] = df["abs_trade"] * df["mid"]
```

### Cell 15: stock, OW
```python
# Set this to False only if you already ran this cell and want to skip re-processing.
RUN_MONTHLY_STANDARDISATION = True

processed_rows = []
all_bin_stock_stats = []
all_fill_stock_stats = []

if RUN_MONTHLY_STANDARDISATION:
    # Process binSamples.
    for raw_path in bin_files:
        month = month_from_filename(raw_path)
        out_path = MONTHLY_DIR / "bin" / f"bin_{month}_standardised.parquet"
        stats_path = MONTHLY_DIR / "bin" / f"bin_{month}_stock_stats.csv"

        print(f"Processing bin {month}: {raw_path.name}")
        raw = read_raw_month(raw_path, kind="bin")
        clean = standardise_intraday_data(raw, kind="bin", month=month, source_file=raw_path.name)
        stats = monthly_stock_stats(clean, kind="bin")
        stats["month"] = month
        stats["parquet_path"] = str(out_path)
        stats["raw_file"] = raw_path.name

        save_parquet(clean, out_path)
        stats.to_csv(stats_path, index=False)

        processed_rows.append({
            "kind": "bin",
            "month": month,
            "raw_file": raw_path.name,
            "raw_size_mb": raw_path.stat().st_size / 1024**2,
            "n_raw_rows": len(raw),
            "n_clean_rows": len(clean),
            "n_rows_dropped_basic_cleaning": clean.attrs.get("n_rows_dropped_basic_cleaning", np.nan),
            "n_stocks": clean["stock"].nunique(),
            "first_datetime": clean["datetime"].min(),
            "last_datetime": clean["datetime"].max(),
            "parquet_path": str(out_path),
            "parquet_size_mb": out_path.stat().st_size / 1024**2,
        })
        all_bin_stock_stats.append(stats)
        del raw, clean, stats
        gc.collect()

    # Process fillSamples.
    for raw_path in fill_files:
        month = month_from_filename(raw_path)
        out_path = MONTHLY_DIR / "fills" / f"fills_{month}_standardised.parquet"
        stats_path = MONTHLY_DIR / "fills" / f"fills_{month}_stock_stats.csv"

        print(f"Processing fills {month}: {raw_path.name}")
        raw = read_raw_month(raw_path, kind="fills")
        clean = standardise_intraday_data(raw, kind="fills", month=month, source_file=raw_path.name)
        stats = monthly_stock_stats(clean, kind="fills")
        stats["month"] = month
        stats["parquet_path"] = str(out_path)
        stats["raw_file"] = raw_path.name

        save_parquet(clean, out_path)
        stats.to_csv(stats_path, index=False)

        processed_rows.append({
            "kind": "fills",
            "month": month,
            "raw_file": raw_path.name,
            "raw_size_mb": raw_path.stat().st_size / 1024**2,
            "n_raw_rows": len(raw),
            "n_clean_rows": len(clean),
            "n_rows_dropped_basic_cleaning": clean.attrs.get("n_rows_dropped_basic_cleaning", np.nan),
            "n_stocks": clean["stock"].nunique(),
            "first_datetime": clean["datetime"].min(),
            "last_datetime": clean["datetime"].max(),
            "parquet_path": str(out_path),
            "parquet_size_mb": out_path.stat().st_size / 1024**2,
        })
        all_fill_stock_stats.append(stats)
        del raw, clean, stats
        gc.collect()

    processed_summary = pd.DataFrame(processed_rows).sort_values(["kind", "month"]).reset_index(drop=True)
    processed_summary.to_csv(REPORT_DIR / "processed_monthly_summary.csv", index=False)
```

### Cell 16: stock
```python
# If this notebook was restarted after processing, reload the saved stats here.
processed_summary = pd.read_csv(REPORT_DIR / "processed_monthly_summary.csv")
bin_stock_month_stats = pd.read_csv(REPORT_DIR / "bin_stock_month_stats.csv")
fill_stock_month_stats = pd.read_csv(REPORT_DIR / "fill_stock_month_stats.csv")

bin_months = sorted(bin_stock_month_stats["month"].astype(str).unique().tolist())
fill_months = sorted(fill_stock_month_stats["month"].astype(str).unique().tolist())

print("Processed bin months:", bin_months)
print("Processed fill months:", fill_months)
```

### Cell 18: rolling, universe, 20, stock, impact, fit, train, test
```python
storage_plan = pd.DataFrame([
    {
        "dataset": "binSamples monthly files",
        "action": "standardise and save one Parquet per month",
        "reason": "Main public binned tape for impact-model fitting; monthly Parquets are faster than raw CSV and preserve chronological traversal.",
    },
    {
        "dataset": "fillSamples monthly files",
        "action": "standardise and save one Parquet per month, but keep separate from binSamples",
        "reason": "Different schema and event-level/fill-level granularity; direct merging with binSamples would create unnecessary duplication and alignment problems.",
    },
    {
        "dataset": "baseline train/test",
        "action": "materialise 20-stock train and test Parquets",
        "reason": "The baseline repeatedly uses exactly the same 20 stocks over two months, so a small materialised subset speeds up Sections 2.2 and 2.3.",
    },
    {
        "dataset": "rolling enhancement",
        "action": "use monthly Parquets plus rolling manifests, not duplicated rolling Parquets",
        "reason": "Rolling over the whole universe would duplicate large files for every consecutive month pair; a manifest tells later code which monthly files and stocks to load.",
    },
])

display(storage_plan)
storage_plan.to_csv(REPORT_DIR / "merge_vs_separate_decision.csv", index=False)
```

### Cell 20: 20, stock, train, test
```python
def parquet_path_for(kind: str, month: str) -> Path:
    if kind == "bin":
        return MONTHLY_DIR / "bin" / f"bin_{month}_standardised.parquet"
    elif kind == "fills":
        return MONTHLY_DIR / "fills" / f"fills_{month}_standardised.parquet"
    else:
        raise ValueError("kind must be 'bin' or 'fills'")


def select_baseline_20_stocks(
    train_month: str,
    test_month: str,
    stats: pd.DataFrame,
    n_stocks: int = 20,
) -> List[str]:
    """Select the top n liquid stocks present in both train and test month."""
    s = stats.copy()
    s["month"] = s["month"].astype(str)
    train = s[s["month"] == str(train_month)].copy()
    test = s[s["month"] == str(test_month)].copy()

    train_stocks = set(train["stock"].astype(str))
    test_stocks = set(test["stock"].astype(str))
    common = train_stocks & test_stocks

    if len(common) < n_stocks:
        raise ValueError(f"Only {len(common)} stocks are common to {train_month} and {test_month}; need {n_stocks}.")

    selected = (
        train[train["stock"].astype(str).isin(common)]
        .sort_values("total_abs_notional", ascending=False)
        .head(n_stocks)["stock"]
        .astype(str)
        .tolist()
    )
    return selected


def load_month_subset(kind: str, month: str, stocks: Optional[List[str]] = None) -> pd.DataFrame:
    path = parquet_path_for(kind, month)
    df = pd.read_parquet(path)
    if stocks is not None:
        df = df[df["stock"].astype(str).isin(stocks)].copy()
    return df.sort_values(["stock", "datetime"]).reset_index(drop=True)
```

### Cell 21: 20, stock, train, test
```python
# Baseline months. By default: first two processed bin months.
# Change these manually only if your group wants a different pair.
BASELINE_TRAIN_MONTH = bin_months[0]
BASELINE_TEST_MONTH = bin_months[1]
N_BASELINE_STOCKS = 20

baseline_stocks = select_baseline_20_stocks(
    BASELINE_TRAIN_MONTH,
    BASELINE_TEST_MONTH,
    bin_stock_month_stats,
    n_stocks=N_BASELINE_STOCKS,
)

print("Baseline train month:", BASELINE_TRAIN_MONTH)
print("Baseline test month: ", BASELINE_TEST_MONTH)
print("Selected 20 stocks: ", baseline_stocks)
```

### Cell 22: 20, stock, train, test
```python
# Load, filter, and save baseline binSamples.
bin_train_20 = load_month_subset("bin", BASELINE_TRAIN_MONTH, baseline_stocks)
bin_test_20 = load_month_subset("bin", BASELINE_TEST_MONTH, baseline_stocks)

assert sorted(bin_train_20["stock"].astype(str).unique()) == sorted(baseline_stocks)
assert sorted(bin_test_20["stock"].astype(str).unique()) == sorted(baseline_stocks)

baseline_bin_train_path = BASELINE_DIR / f"bin_train_{BASELINE_TRAIN_MONTH}_20stocks.parquet"
baseline_bin_test_path = BASELINE_DIR / f"bin_test_{BASELINE_TEST_MONTH}_20stocks.parquet"
save_parquet(bin_train_20, baseline_bin_train_path)
save_parquet(bin_test_20, baseline_bin_test_path)

print("Saved:", baseline_bin_train_path)
print("Saved:", baseline_bin_test_path)
print("Train shape:", bin_train_20.shape)
print("Test shape: ", bin_test_20.shape)
```

### Cell 23: universe, 20, stock, OW, train, test
```python
# Prepare fillSamples for the same two baseline months.
# We do NOT force fillSamples to have the same 20-stock universe, because in this dataset fills may cover fewer stocks.
# We simply filter to baseline stocks when they exist and save the available rows.

fill_train_path = parquet_path_for("fills", BASELINE_TRAIN_MONTH) if BASELINE_TRAIN_MONTH in fill_months else None
fill_test_path = parquet_path_for("fills", BASELINE_TEST_MONTH) if BASELINE_TEST_MONTH in fill_months else None

baseline_fill_outputs = {}

if fill_train_path is not None and fill_train_path.exists():
    fills_train = load_month_subset("fills", BASELINE_TRAIN_MONTH, baseline_stocks)
    out = BASELINE_DIR / f"fills_train_{BASELINE_TRAIN_MONTH}_available_baseline_stocks.parquet"
    save_parquet(fills_train, out)
    baseline_fill_outputs["fills_train_path"] = str(out)
    print("Saved:", out, "shape:", fills_train.shape, "stocks:", sorted(fills_train["stock"].astype(str).unique()))
else:
    fills_train = pd.DataFrame()
    baseline_fill_outputs["fills_train_path"] = None
    print("No fill file available for train month.")

if fill_test_path is not None and fill_test_path.exists():
    fills_test = load_month_subset("fills", BASELINE_TEST_MONTH, baseline_stocks)
    out = BASELINE_DIR / f"fills_test_{BASELINE_TEST_MONTH}_available_baseline_stocks.parquet"
    save_parquet(fills_test, out)
    baseline_fill_outputs["fills_test_path"] = str(out)
    print("Saved:", out, "shape:", fills_test.shape, "stocks:", sorted(fills_test["stock"].astype(str).unique()))
else:
    fills_test = pd.DataFrame()
    baseline_fill_outputs["fills_test_path"] = None
    print("No fill file available for test month.")
```

### Cell 24: 20, stock, OW, train, test
```python
# Baseline summary table for the report.
baseline_stats = bin_stock_month_stats[
    (bin_stock_month_stats["month"].astype(str).isin([BASELINE_TRAIN_MONTH, BASELINE_TEST_MONTH]))
    & (bin_stock_month_stats["stock"].astype(str).isin(baseline_stocks))
].copy()

baseline_stats = baseline_stats[[
    "month", "stock", "n_rows", "n_trading_days", "total_abs_trade", "total_abs_notional",
    "mean_mid", "median_spread_bps", "median_depth"
]].sort_values(["month", "total_abs_notional"], ascending=[True, False])

baseline_stats.to_csv(BASELINE_DIR / "baseline_20stocks_summary.csv", index=False)
display(baseline_stats.head(40))
```

### Cell 25: in-sample, 20, stock, train, test
```python
# Save full baseline configuration.
baseline_config = {
    "section": "2.1 Data Preparation",
    "baseline_train_month": BASELINE_TRAIN_MONTH,
    "baseline_test_month": BASELINE_TEST_MONTH,
    "n_baseline_stocks": N_BASELINE_STOCKS,
    "baseline_stocks": baseline_stocks,
    "bin_train_path": str(baseline_bin_train_path),
    "bin_test_path": str(baseline_bin_test_path),
    **baseline_fill_outputs,
    "selection_rule": "Top 20 stocks by in-sample total_abs_notional among stocks present in both train and test months.",
    "bin_and_fills_merged": False,
    "reason_not_merged": "binSamples and fillSamples have different granularity and schemas; they are saved separately.",
}

with open(BASELINE_DIR / "section_2_1_baseline_config.json", "w") as f:
    json.dump(baseline_config, f, indent=2)

print(json.dumps(baseline_config, indent=2))
```

### Cell 27: rolling, train, test
```python
def build_consecutive_month_pairs(months: List[str]) -> List[Tuple[str, str]]:
    months = sorted([str(m) for m in months])
    return list(zip(months[:-1], months[1:]))

rolling_pairs = build_consecutive_month_pairs(bin_months)
print("Rolling pairs:")
for tr, te in rolling_pairs:
    print(f"  {tr} -> {te}")

assert len(rolling_pairs) >= 1, "Need at least two months for rolling train/test pairs."
```

### Cell 28: rolling, universe, 20, stock, OW, train, test
```python
# Build rolling pair summary and universe tables.
rolling_pair_rows = []
rolling_universe_rows = []

stats = bin_stock_month_stats.copy()
stats["month"] = stats["month"].astype(str)
stats["stock"] = stats["stock"].astype(str)

for pair_id, (train_month, test_month) in enumerate(rolling_pairs, start=1):
    train_stats = stats[stats["month"] == train_month].copy()
    test_stats = stats[stats["month"] == test_month].copy()

    train_stocks = set(train_stats["stock"])
    test_stocks = set(test_stats["stock"])
    common_stocks = sorted(train_stocks & test_stocks)

    # Merge stock-level metrics for the common universe.
    pair_universe = (
        train_stats[train_stats["stock"].isin(common_stocks)][["stock", "total_abs_notional", "n_rows", "n_trading_days"]]
        .rename(columns={
            "total_abs_notional": "train_total_abs_notional",
            "n_rows": "train_n_rows",
            "n_trading_days": "train_n_trading_days",
        })
        .merge(
            test_stats[test_stats["stock"].isin(common_stocks)][["stock", "total_abs_notional", "n_rows", "n_trading_days"]]
            .rename(columns={
                "total_abs_notional": "test_total_abs_notional",
                "n_rows": "test_n_rows",
                "n_trading_days": "test_n_trading_days",
            }),
            on="stock",
            how="inner",
        )
    )
    pair_universe["pair_id"] = pair_id
    pair_universe["train_month"] = train_month
    pair_universe["test_month"] = test_month
    pair_universe = pair_universe.sort_values("train_total_abs_notional", ascending=False)
    rolling_universe_rows.append(pair_universe)

    rolling_pair_rows.append({
        "pair_id": pair_id,
        "train_month": train_month,
        "test_month": test_month,
        "train_bin_path": str(parquet_path_for("bin", train_month)),
        "test_bin_path": str(parquet_path_for("bin", test_month)),
        "train_fill_path": str(parquet_path_for("fills", train_month)) if train_month in fill_months else None,
        "test_fill_path": str(parquet_path_for("fills", test_month)) if test_month in fill_months else None,
        "n_train_stocks": len(train_stocks),
        "n_test_stocks": len(test_stocks),
        "n_common_stocks_pair_universe": len(common_stocks),
        "top20_common_stocks_by_train_notional": pair_universe.head(20)["stock"].tolist(),
    })

rolling_pair_summary = pd.DataFrame(rolling_pair_rows)
rolling_universe_by_pair = pd.concat(rolling_universe_rows, ignore_index=True)

rolling_pair_summary.to_csv(ROLLING_DIR / "rolling_pair_summary.csv", index=False)
rolling_universe_by_pair.to_csv(ROLLING_DIR / "rolling_universe_by_pair.csv", index=False)

print("Rolling pair summary:")
display(rolling_pair_summary)

print("Rolling universe example:")
display(rolling_universe_by_pair.head(30))
```

### Cell 29: rolling, universe, stock
```python
# Also compute the strict global common universe: stocks present in every bin month.
# This is not the default rolling universe, but it is useful as a stricter robustness check.
sets_by_month = {
    m: set(stats.loc[stats["month"] == m, "stock"].astype(str))
    for m in bin_months
}

global_common_universe = sorted(set.intersection(*sets_by_month.values())) if sets_by_month else []

global_common_df = pd.DataFrame({"stock": global_common_universe})
global_common_df.to_csv(ROLLING_DIR / "global_common_universe_all_months.csv", index=False)

print(f"Number of stocks present in every bin month: {len(global_common_universe)}")
display(global_common_df.head(50))
```

### Cell 30: in-sample, out-of-sample, rolling, universe, stock, train, test
```python
# Save an overall rolling config for later sections.
rolling_config = {
    "section": "2.1 Data Preparation rolling enhancement",
    "rolling_rule": "For each consecutive month pair, use train_month as in-sample and test_month as out-of-sample.",
    "universe_rule": "Use all stocks common to the train and test month of that pair.",
    "data_storage_rule": "Use standardised monthly Parquet files plus rolling manifests; do not duplicate full rolling datasets.",
    "rolling_pair_summary_path": str(ROLLING_DIR / "rolling_pair_summary.csv"),
    "rolling_universe_by_pair_path": str(ROLLING_DIR / "rolling_universe_by_pair.csv"),
    "global_common_universe_path": str(ROLLING_DIR / "global_common_universe_all_months.csv"),
    "monthly_bin_parquet_dir": str(MONTHLY_DIR / "bin"),
    "monthly_fill_parquet_dir": str(MONTHLY_DIR / "fills"),
}

with open(ROLLING_DIR / "section_2_1_rolling_config.json", "w") as f:
    json.dump(rolling_config, f, indent=2)

print(json.dumps(rolling_config, indent=2))
```

### Cell 32: 20, stock, train, test
```python
# Baseline loading example for Section 2.2 / 2.3.
baseline_train = pd.read_parquet(BASELINE_DIR / f"bin_train_{BASELINE_TRAIN_MONTH}_20stocks.parquet")
baseline_test = pd.read_parquet(BASELINE_DIR / f"bin_test_{BASELINE_TEST_MONTH}_20stocks.parquet")

print("Baseline train:", baseline_train.shape, baseline_train["datetime"].min(), "to", baseline_train["datetime"].max())
print("Baseline test: ", baseline_test.shape, baseline_test["datetime"].min(), "to", baseline_test["datetime"].max())
print("Baseline train stocks:", sorted(baseline_train["stock"].astype(str).unique()))
print("Baseline test stocks: ", sorted(baseline_test["stock"].astype(str).unique()))
```

### Cell 33: rolling, universe, stock, train, test
```python
# Rolling loading example: first rolling pair.
pair_summary = pd.read_csv(ROLLING_DIR / "rolling_pair_summary.csv")
pair_universe = pd.read_csv(ROLLING_DIR / "rolling_universe_by_pair.csv")

first_pair = pair_summary.iloc[0]
first_pair_id = int(first_pair["pair_id"])
first_pair_stocks = pair_universe.loc[pair_universe["pair_id"] == first_pair_id, "stock"].astype(str).tolist()

rolling_train_month = str(first_pair["train_month"])
rolling_test_month = str(first_pair["test_month"])

rolling_train = load_month_subset("bin", rolling_train_month, first_pair_stocks)
rolling_test = load_month_subset("bin", rolling_test_month, first_pair_stocks)

print(f"Rolling pair {first_pair_id}: {rolling_train_month} -> {rolling_test_month}")
print("Number of stocks:", len(first_pair_stocks))
print("Rolling train shape:", rolling_train.shape)
print("Rolling test shape: ", rolling_test.shape)
```

### Cell 35: 20, OW
```python
# Show output tree.
for root, dirs, files in os.walk(PROCESSED_BASE):
    level = root.replace(str(PROCESSED_BASE), "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{Path(root).name}/")
    subindent = "  " * (level + 1)
    for f in sorted(files)[:20]:
        print(f"{subindent}{f}")
    if len(files) > 20:
        print(f"{subindent}... {len(files)-20} more files")
```

### Cell 37: rolling, universe, impact
```python
import shutil
from pathlib import Path

DRIVE_OUTPUT_BASE = Path("/content/drive/MyDrive/Quantitative Trading and Price Impact/project_data")
DRIVE_PROCESSED_21 = DRIVE_OUTPUT_BASE / "processed_2_1"

DRIVE_PROCESSED_21.mkdir(parents=True, exist_ok=True)

shutil.copytree(
    PROCESSED_BASE,
    DRIVE_PROCESSED_21,
    dirs_exist_ok=True
)

print("Copied Section 2.1 outputs to Drive:")
print(DRIVE_PROCESSED_21)

for p in (DRIVE_PROCESSED_21 / "rolling_full_universe").glob("*"):
    print(p.name)
```

## notebooks/2_2_rolling_baseline_no_enhancement.ipynb

### Cell 3: 20, OW
```python
!pip -q install pyarrow

import os
import gc
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

pd.set_option("display.max_columns", 160)
pd.set_option("display.width", 200)
```

### Cell 5: rolling, universe, missing, impact
```python
import shutil
from pathlib import Path

# Local Colab working location
LOCAL_BASE = Path("/content/project_data")

# Permanent Drive backup location
DRIVE_OUTPUT_BASE = Path("/content/drive/MyDrive/Quantitative Trading and Price Impact/project_data")
DRIVE_PROCESSED_21 = DRIVE_OUTPUT_BASE / "processed_2_1"

# Section 2.1 local output locations
SECTION21_DIR = LOCAL_BASE / "processed_2_1"
MONTHLY21_DIR = SECTION21_DIR / "monthly_standardised"
ROLLING21_DIR = SECTION21_DIR / "rolling_full_universe"

# If Section 2.1 outputs are missing locally, restore them from Drive
if not ROLLING21_DIR.exists():
    if DRIVE_PROCESSED_21.exists():
        print("Restoring Section 2.1 outputs from Drive to local Colab storage...")
        shutil.copytree(DRIVE_PROCESSED_21, SECTION21_DIR, dirs_exist_ok=True)
    else:
        raise FileNotFoundError(
            f"Section 2.1 outputs are missing locally and on Drive.\n"
            f"Expected Drive backup here: {DRIVE_PROCESSED_21}\n"
            f"Go back to 2.1 and run the Drive-copy cell first."
        )

# Section 2.2 output locations
SECTION22_DIR = LOCAL_BASE / "processed_2_2_rolling_baseline"
PARAM_DIR = SECTION22_DIR / "parameters"
METRIC_DIR = SECTION22_DIR / "metrics"
REPORT_DIR = SECTION22_DIR / "report_tables"
FIG_DIR = SECTION22_DIR / "figures"
LOG_DIR = SECTION22_DIR / "logs"

for p in [SECTION22_DIR, PARAM_DIR, METRIC_DIR, REPORT_DIR, FIG_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("Section 2.1 rolling folder:", ROLLING21_DIR)
print("Section 2.1 monthly folder: ", MONTHLY21_DIR)
print("Section 2.2 output folder:  ", SECTION22_DIR)

print("\nFiles in rolling folder:")
for p in ROLLING21_DIR.glob("*"):
    print(p.name)
```

### Cell 7: rolling, universe, stock, OW, train, test
```python
pair_summary_path = ROLLING21_DIR / "rolling_pair_summary.csv"
universe_path = ROLLING21_DIR / "rolling_universe_by_pair.csv"
rolling_config_path = ROLLING21_DIR / "section_2_1_rolling_config.json"

if not pair_summary_path.exists() or not universe_path.exists():
    raise FileNotFoundError(
        "Could not find Section 2.1 rolling outputs. "
        "Run 2_1_full_as_spec_completed.ipynb first, including the rolling enhancement cells."
    )

pair_summary = pd.read_csv(pair_summary_path)
pair_universe = pd.read_csv(universe_path)

with open(rolling_config_path, "r") as f:
    section21_rolling_config = json.load(f)

pair_summary["train_month"] = pair_summary["train_month"].astype(str)
pair_summary["test_month"] = pair_summary["test_month"].astype(str)
pair_universe["train_month"] = pair_universe["train_month"].astype(str)
pair_universe["test_month"] = pair_universe["test_month"].astype(str)
pair_universe["stock"] = pair_universe["stock"].astype(str)

print("Rolling pairs:")
display(pair_summary)

print("Universe rows:")
display(pair_universe.head())

print("Stocks per pair:")
display(pair_universe.groupby("pair_id")["stock"].nunique().reset_index(name="n_stocks"))
```

### Cell 8: rolling, universe, 20, stock
```python
# Safety check: this notebook must continue from the enhanced rolling structure, not the 20-stock baseline.
n_stocks_by_pair = pair_universe.groupby("pair_id")["stock"].nunique()
print(n_stocks_by_pair.describe())

assert len(pair_summary) >= 1, "Need at least one rolling pair."
assert n_stocks_by_pair.min() > 20, (
    "The rolling universe has 20 or fewer stocks in at least one pair. "
    "Check that you are using the full rolling-universe output from Section 2.1, not the 20-stock baseline."
)

print("Confirmed: using rolling full-universe pairs, not the fixed 20-stock baseline.")
```

### Cell 10: universe, 20, stock, OW, reduced, fit, OLS, train, test
```python
# Main run controls.
# Leave these as None for the final project run.
RUN_PAIR_IDS: Optional[List[int]] = None       # example for debugging only: [1]
MAX_STOCKS_PER_PAIR: Optional[int] = None     # keep None for full universe; do not set to 20

# Minimum data needed to fit a stock-level model.
MIN_TRAIN_ROWS_PER_STOCK = 100
MIN_TEST_ROWS_PER_STOCK = 20

# OW half-life candidates in seconds.
OW_HALF_LIFE_GRID_SEC = [30, 60, 120, 300, 600, 1200, 1800]

# Reduced-form model features.
REDUCED_FORM_FEATURES = [
    "x_flow",
    "x_trade",
    "x_hidden",
    "x_flow_depth",
    "lobImb",
    "effLobImb",
    "spread_bps",
]

# Columns needed from the Section 2.1 monthly standardised binSamples.
BIN_COLUMNS_NEEDED = [
    "date", "time", "stock", "trade", "orderFlow", "hidden", "auction",
    "mid", "midEnd", "spread", "lobImb", "effLobImb", "depth",
    "datetime", "trading_date", "seconds_from_open"
]
```

### Cell 12: rolling, universe, 20, stock, missing, OW, OLS
```python
def monthly_bin_path(month: str) -> Path:
    """Return the Section 2.1 standardised binSamples Parquet path for one month."""
    return MONTHLY21_DIR / "bin" / f"bin_{str(month)}_standardised.parquet"


def load_bin_month_for_universe(month: str, stocks: List[str], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load one standardised binSamples month and filter to the requested stock universe.

    This uses the monthly Parquet from Section 2.1. It does not select top-20 stocks.
    """
    path = monthly_bin_path(month)
    if not path.exists():
        raise FileNotFoundError(f"Missing monthly bin Parquet for {month}: {path}")

    use_cols = columns if columns is not None else None

    # Try Parquet predicate pushdown. If it is unavailable for this file, fall back safely.
    try:
        df = pd.read_parquet(path, columns=use_cols, filters=[("stock", "in", list(stocks))])
    except Exception:
        df = pd.read_parquet(path, columns=use_cols)
        df["stock"] = df["stock"].astype(str)
        df = df[df["stock"].isin(set(stocks))].copy()

    df["stock"] = df["stock"].astype(str)
    return df.reset_index(drop=True)



def get_pair_stock_universe(pair_id: int) -> List[str]:
    """Return the full common stock universe for a rolling pair."""
    stocks = (
        pair_universe.loc[pair_universe["pair_id"] == pair_id, "stock"]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if MAX_STOCKS_PER_PAIR is not None:
        # This is only for debugging. Do not use this for the final run.
        stocks = stocks[:MAX_STOCKS_PER_PAIR]
    return stocks
```

### Cell 14: stock, fillna, OW, fit, OLS, train, test
```python
def make_seconds_from_open(df: pd.DataFrame) -> pd.Series:
    """Create seconds since 09:30 when Section 2.1 did not already save it."""
    dt = pd.to_datetime(df["datetime"])
    open_time = pd.to_datetime(dt.dt.date.astype(str) + " 09:30:00")
    return (dt - open_time).dt.total_seconds().astype("float64")


def compute_training_scales(train_df: pd.DataFrame) -> Dict:
    """
    Compute stock-level training scales used to normalise order-flow variables.

    These are computed from the train month only and then reused for the test month.
    """
    tmp = train_df.copy()
    tmp["stock"] = tmp["stock"].astype(str)
    tmp["trading_date"] = tmp["trading_date"].astype(str)
    tmp["abs_orderFlow"] = pd.to_numeric(tmp["orderFlow"], errors="coerce").abs()

    daily_abs_flow = (
        tmp.groupby(["stock", "trading_date"], observed=True)["abs_orderFlow"]
        .sum()
        .reset_index(name="daily_abs_orderFlow")
    )

    stock_flow_scale = (
        daily_abs_flow.groupby("stock", observed=True)["daily_abs_orderFlow"]
        .median()
        .replace(0, np.nan)
    )

    global_flow_scale = float(np.nanmedian(stock_flow_scale.values))
    if not np.isfinite(global_flow_scale) or global_flow_scale <= 0:
        global_flow_scale = 1.0

    stock_flow_scale = stock_flow_scale.fillna(global_flow_scale).to_dict()

    median_depth = (
        tmp.groupby("stock", observed=True)["depth"]
        .median()
        .replace(0, np.nan)
    )

    global_depth_scale = float(np.nanmedian(median_depth.values))
    if not np.isfinite(global_depth_scale) or global_depth_scale <= 0:
        global_depth_scale = 1.0

    median_depth = median_depth.fillna(global_depth_scale).to_dict()

    return {
        "stock_flow_scale": {str(k): float(v) for k, v in stock_flow_scale.items()},
        "global_flow_scale": float(global_flow_scale),
        "stock_depth_scale": {str(k): float(v) for k, v in median_depth.items()},
        "global_depth_scale": float(global_depth_scale),
    }


def prepare_model_frame(df: pd.DataFrame, scales: Dict) -> pd.DataFrame:
    """Create the modelling variables used by both baseline Section 2.2 models."""
    out = df.copy()

    out["stock"] = out["stock"].astype(str)
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["trading_date"] = out["trading_date"].astype(str)

    if "seconds_from_open" not in out.columns:
        out["seconds_from_open"] = make_seconds_from_open(out)

    # Minimal required filtering for model fitting.
    out = out[out["datetime"].notna()].copy()
    out = out[(out["mid"] > 0) & (out["midEnd"] > 0)].copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    # Target: within-bin mid-price move in basis points.
    out["ret_bps"] = 10000.0 * (out["midEnd"] - out["mid"]) / out["mid"]

    # Stock-level flow normalisation from the train month.
    flow_scale_map = scales["stock_flow_scale"]
    depth_scale_map = scales["stock_depth_scale"]

    out["flow_scale"] = out["stock"].map(flow_scale_map).fillna(scales["global_flow_scale"]).astype(float)
```

### Cell 16: fillna, dropna, beta, OW, sqrt, fit, OLS, regression
```python
def fit_ols(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> Dict:
    """Fit y = intercept + X beta by least squares."""
    use_cols = [y_col] + x_cols
    d = df[use_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    if len(d) < len(x_cols) + 5:
        raise ValueError(f"Not enough rows to fit OLS with features {x_cols}. Rows={len(d)}")

    y = d[y_col].to_numpy(dtype=float)
    X = d[x_cols].to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])

    coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)

    coef_dict = {"intercept": float(coef[0])}
    coef_dict.update({name: float(value) for name, value in zip(x_cols, coef[1:])})

    return {"coef": coef_dict}


def predict_ols(df: pd.DataFrame, coef: Dict, x_cols: List[str]) -> np.ndarray:
    """Predict using a coefficient dictionary produced by fit_ols."""
    X = df[x_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    pred = np.full(len(df), float(coef.get("intercept", 0.0)))
    for j, c in enumerate(x_cols):
        pred += float(coef.get(c, 0.0)) * X[:, j]
    return pred


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return standard regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "n": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "corr": np.nan,
            "mean_error": np.nan,
        }

    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = np.nan

    return {
        "n": int(len(y_true)),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "corr": corr,
        "mean_error": float(np.mean(err)),
    }


class MetricAccumulator:
    """Accumulate overall metrics without storing all row-level predictions."""
    def __init__(self):
        self.n = 0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.sum_pred = 0.0
        self.sum_pred2 = 0.0
        self.sum_y_pred = 0.0
        self.sum_err = 0.0
```

### Cell 18: stock, OW, fit, OLS, regression, train
```python
def add_ow_state_one_stock(df: pd.DataFrame, half_life_sec: float, flow_col: str = "x_flow") -> pd.DataFrame:
    """Add the pre-trade OW state for one stock, with daily resets."""
    out = df.sort_values(["trading_date", "datetime"]).reset_index(drop=True).copy()

    state_pre = np.zeros(len(out), dtype=float)
    state_post = np.zeros(len(out), dtype=float)

    for _, positions in out.groupby("trading_date", observed=True, sort=False).indices.items():
        positions = np.asarray(positions, dtype=int)
        sub = out.iloc[positions]
        seconds = sub["seconds_from_open"].to_numpy(dtype=float)
        flows = sub[flow_col].to_numpy(dtype=float)

        prev_state = 0.0
        prev_sec = None

        for k, pos in enumerate(positions):
            if prev_sec is None:
                dt = 0.0
            else:
                dt = max(float(seconds[k] - prev_sec), 0.0)

            phi = math.exp(-dt / half_life_sec) if half_life_sec > 0 else 0.0
            pre = phi * prev_state
            post = pre + flows[k]

            state_pre[pos] = pre
            state_post[pos] = post

            prev_state = post
            prev_sec = seconds[k]

    out["ow_state_pre"] = state_pre
    out["ow_state_post"] = state_post
    return out


def fit_ow_for_stock(train_stock: pd.DataFrame, half_life_grid: List[float]) -> Dict:
    """Choose the OW half-life by train RMSE and fit the final OW coefficients."""
    best = None
    x_cols = ["x_flow", "ow_state_pre"]

    for hl in half_life_grid:
        with_state = add_ow_state_one_stock(train_stock, half_life_sec=hl)
        fit = fit_ols(with_state, "ret_bps", x_cols)
        pred = predict_ols(with_state, fit["coef"], x_cols)
        metrics = regression_metrics(with_state["ret_bps"], pred)

        candidate = {
            "half_life_sec": float(hl),
            "coef": fit["coef"],
            "metrics": metrics,
        }

        if best is None or candidate["metrics"]["rmse"] < best["metrics"]["rmse"]:
            best = candidate

    return best
```

### Cell 20: stock, reduced, fit, OLS, train
```python
def fit_reduced_form_for_stock(train_stock: pd.DataFrame) -> Dict:
    """Fit the reduced-form parametric model for one stock."""
    fit = fit_ols(train_stock, "ret_bps", REDUCED_FORM_FEATURES)
    return {"coef": fit["coef"]}
```

### Cell 22: rolling, universe, stock, OW, reduced, impact, fit, OLS, regression, train, test
```python
def process_one_pair(pair_row: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit OW and reduced-form baseline models for one rolling pair.

    Returns:
    - parameter rows
    - stock-level metric rows
    - pair-level overall metric rows
    """
    pair_id = int(pair_row["pair_id"])
    train_month = str(pair_row["train_month"])
    test_month = str(pair_row["test_month"])

    stocks = get_pair_stock_universe(pair_id)
    print(f"\nPair {pair_id}: {train_month} -> {test_month}")
    print(f"Full common universe size: {len(stocks)}")

    if len(stocks) == 0:
        raise ValueError(f"No stocks found for pair_id={pair_id}")

    train_raw = load_bin_month_for_universe(train_month, stocks, columns=BIN_COLUMNS_NEEDED)
    test_raw = load_bin_month_for_universe(test_month, stocks, columns=BIN_COLUMNS_NEEDED)

    print("Loaded train:", train_raw.shape)
    print("Loaded test: ", test_raw.shape)

    scales = compute_training_scales(train_raw)
    train = prepare_model_frame(train_raw, scales)
    test = prepare_model_frame(test_raw, scales)

    common_after_clean = sorted(set(train["stock"].unique()) & set(test["stock"].unique()) & set(stocks))
    print("Common stocks after minimal model cleaning:", len(common_after_clean))

    param_rows = []
    metric_rows = []
    failure_rows = []

    overall_acc = {
        ("OW_transient", "train"): MetricAccumulator(),
        ("OW_transient", "test"): MetricAccumulator(),
        ("reduced_form", "train"): MetricAccumulator(),
        ("reduced_form", "test"): MetricAccumulator(),
    }

    train_groups = {str(k): v.reset_index(drop=True) for k, v in train.groupby("stock", observed=True, sort=False)}
    test_groups = {str(k): v.reset_index(drop=True) for k, v in test.groupby("stock", observed=True, sort=False)}

    for j, stock in enumerate(common_after_clean, start=1):
        if j % 25 == 0 or j == 1 or j == len(common_after_clean):
            print(f"  fitting stock {j}/{len(common_after_clean)}: {stock}")

        tr = train_groups.get(stock, pd.DataFrame())
        te = test_groups.get(stock, pd.DataFrame())

        if len(tr) < MIN_TRAIN_ROWS_PER_STOCK or len(te) < MIN_TEST_ROWS_PER_STOCK:
            failure_rows.append({
                "pair_id": pair_id,
                "train_month": train_month,
                "test_month": test_month,
                "stock": stock,
                "reason": "insufficient_rows",
                "train_rows": len(tr),
                "test_rows": len(te),
            })
            continue

        # Model 1: OW transient impact.
        try:
            ow_fit = fit_ow_for_stock(tr, OW_HALF_LIFE_GRID_SEC)
            ow_coef = ow_fit["coef"]
            ow_hl = ow_fit["half_life_sec"]

            tr_ow = add_ow_state_one_stock(tr, ow_hl)
            te_ow = add_ow_state_one_stock(te, ow_hl)

            ow_features = ["x_flow", "ow_state_pre"]
            tr_pred = predict_ols(tr_ow, ow_coef, ow_features)
            te_pred = predict_ols(te_ow, ow_coef, ow_features)

            tr_metrics = regression_metrics(tr_ow["ret_bps"], tr_pred)
```

### Cell 23: rolling, stock, OW, fit, train, test
```python
# Select rolling pairs to run.
pairs_to_run = pair_summary.copy()

if RUN_PAIR_IDS is not None:
    pairs_to_run = pairs_to_run[pairs_to_run["pair_id"].isin(RUN_PAIR_IDS)].copy()

print("Pairs to run:")
display(pairs_to_run[["pair_id", "train_month", "test_month"]])

all_param_parts = []
all_metric_parts = []
all_overall_parts = []

for _, pair_row in pairs_to_run.iterrows():
    params_df, metrics_df, overall_df = process_one_pair(pair_row)

    all_param_parts.append(params_df)
    all_metric_parts.append(metrics_df)
    all_overall_parts.append(overall_df)

    # Save combined outputs after every pair.
    combined_params = pd.concat(all_param_parts, ignore_index=True) if all_param_parts else pd.DataFrame()
    combined_metrics = pd.concat(all_metric_parts, ignore_index=True) if all_metric_parts else pd.DataFrame()
    combined_overall = pd.concat(all_overall_parts, ignore_index=True) if all_overall_parts else pd.DataFrame()

    combined_params.to_csv(PARAM_DIR / "section_2_2_rolling_baseline_params_all_pairs.csv", index=False)
    combined_metrics.to_csv(METRIC_DIR / "section_2_2_rolling_baseline_stock_metrics_all_pairs.csv", index=False)
    combined_overall.to_csv(METRIC_DIR / "section_2_2_rolling_baseline_overall_metrics_all_pairs.csv", index=False)

print("Rolling Section 2.2 fitting complete.")
```

### Cell 25: rolling, stock, OW
```python
params_all = pd.read_csv(PARAM_DIR / "section_2_2_rolling_baseline_params_all_pairs.csv")
stock_metrics_all = pd.read_csv(METRIC_DIR / "section_2_2_rolling_baseline_stock_metrics_all_pairs.csv")
overall_metrics_all = pd.read_csv(METRIC_DIR / "section_2_2_rolling_baseline_overall_metrics_all_pairs.csv")

print("All parameter rows:", params_all.shape)
display(params_all.head())

print("Overall rolling metrics:")
display(overall_metrics_all)

print("Stock-level metrics sample:")
display(stock_metrics_all.head())
```

### Cell 26: stock, OW, reduced
```python
# Separate parameter files by model for easier Section 2.3 loading.
ow_params = params_all[params_all["model"] == "OW_transient"].copy()
rf_params = params_all[params_all["model"] == "reduced_form"].copy()

ow_params.to_csv(PARAM_DIR / "ow_transient_params_by_pair_stock.csv", index=False)
rf_params.to_csv(PARAM_DIR / "reduced_form_params_by_pair_stock.csv", index=False)

print("OW params:", ow_params.shape)
print("Reduced-form params:", rf_params.shape)
```

### Cell 28: out-of-sample, rolling, OW, test
```python
# Table: average out-of-sample performance across rolling pairs.
test_overall = overall_metrics_all[overall_metrics_all["sample"] == "test"].copy()

summary_by_model = (
    test_overall.groupby("model", observed=True)
    .agg(
        n_pairs=("pair_id", "nunique"),
        total_test_rows=("n", "sum"),
        avg_test_rmse=("rmse", "mean"),
        median_test_rmse=("rmse", "median"),
        avg_test_mae=("mae", "mean"),
        avg_test_r2=("r2", "mean"),
        avg_test_corr=("corr", "mean"),
    )
    .reset_index()
)

summary_by_model.to_csv(REPORT_DIR / "section_2_2_rolling_baseline_summary_by_model.csv", index=False)
display(summary_by_model)
```

### Cell 29: out-of-sample, train, test
```python
# Table: pair-by-pair out-of-sample comparison.
pair_model_comparison = (
    test_overall[["pair_id", "train_month", "test_month", "model", "n", "rmse", "mae", "r2", "corr"]]
    .sort_values(["pair_id", "model"])
    .reset_index(drop=True)
)

pair_model_comparison.to_csv(REPORT_DIR / "section_2_2_pair_model_comparison.csv", index=False)
display(pair_model_comparison)
```

### Cell 30: out-of-sample, rolling, pivot, OW, test
```python
# Figure 1: out-of-sample RMSE by rolling pair.
rmse_pivot = test_overall.pivot(index="pair_id", columns="model", values="rmse").sort_index()

ax = rmse_pivot.plot(kind="bar", figsize=(12, 5))
ax.set_title("Section 2.2 rolling baseline: out-of-sample RMSE by pair")
ax.set_xlabel("Rolling pair id")
ax.set_ylabel("RMSE of within-bin mid-price move, bps")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "rolling_test_rmse_by_pair.png", dpi=150)
plt.show()
```

### Cell 31: out-of-sample, rolling, pivot, OW, test
```python
# Figure 2: out-of-sample correlation by rolling pair.
corr_pivot = test_overall.pivot(index="pair_id", columns="model", values="corr").sort_index()

ax = corr_pivot.plot(kind="bar", figsize=(12, 5))
ax.set_title("Section 2.2 rolling baseline: out-of-sample prediction correlation by pair")
ax.set_xlabel("Rolling pair id")
ax.set_ylabel("Correlation(actual, predicted)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "rolling_test_corr_by_pair.png", dpi=150)
plt.show()
```

### Cell 32: rolling, stock, dropna, OW, fit
```python
# Figure 3: fitted OW half-life distribution.
if "half_life_sec" in ow_params.columns and len(ow_params) > 0:
    ax = ow_params["half_life_sec"].dropna().plot(kind="hist", bins=len(OW_HALF_LIFE_GRID_SEC), figsize=(9, 5))
    ax.set_title("Distribution of selected OW half-lives across rolling pairs and stocks")
    ax.set_xlabel("Half-life, seconds")
    ax.set_ylabel("Number of fitted stock-pair models")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ow_half_life_distribution.png", dpi=150)
    plt.show()
else:
    print("No OW half-life values available to plot.")
```

### Cell 34: out-of-sample, rolling, universe, 20, stock, OW, reduced, impact, fit, train
```python
handoff = {
    "section": "2.2 Impact Model Fitting rolling-universe baseline",
    "continues_from": "2.1 rolling_full_universe enhancement",
    "uses_20_stock_baseline": False,
    "uses_full_rolling_universe": True,
    "section_2_2_enhancement_used": False,
    "models_fitted": ["OW_transient", "reduced_form"],
    "no_enhancement_details": [
        "No non-parametric model.",
        "No regularisation.",
        "No third sample.",
        "No regularisation hyperparameter tuning.",
    ],
    "input_files": {
        "rolling_pair_summary": str(pair_summary_path),
        "rolling_universe_by_pair": str(universe_path),
        "monthly_bin_parquet_dir": str(MONTHLY21_DIR / "bin"),
    },
    "parameter_files": {
        "all_params": str(PARAM_DIR / "section_2_2_rolling_baseline_params_all_pairs.csv"),
        "ow_transient": str(PARAM_DIR / "ow_transient_params_by_pair_stock.csv"),
        "reduced_form": str(PARAM_DIR / "reduced_form_params_by_pair_stock.csv"),
    },
    "metric_files": {
        "stock_metrics": str(METRIC_DIR / "section_2_2_rolling_baseline_stock_metrics_all_pairs.csv"),
        "overall_metrics": str(METRIC_DIR / "section_2_2_rolling_baseline_overall_metrics_all_pairs.csv"),
        "summary_by_model": str(REPORT_DIR / "section_2_2_rolling_baseline_summary_by_model.csv"),
    },
    "model_assumptions": [
        "Models are fitted stock by stock within each rolling pair.",
        "Only the train month is used for fitting.",
        "The next month is used for out-of-sample evaluation.",
        "OW transient state is reset at the start of each trading day.",
        "Order flow is normalised using train-month stock-level median daily absolute order flow.",
    ],
}

with open(SECTION22_DIR / "section_2_2_rolling_baseline_handoff.json", "w") as f:
    json.dump(handoff, f, indent=2)

print(json.dumps(handoff, indent=2))
```

### Cell 36: rolling, impact
```python
import shutil
from pathlib import Path

LOCAL_PROCESSED_22 = Path("/content/project_data/processed_2_2_rolling_baseline")

DRIVE_PROCESSED_22 = Path(
    "/content/drive/MyDrive/Quantitative Trading and Price Impact/project_data/processed_2_2_rolling_baseline"
)

DRIVE_PROCESSED_22.mkdir(parents=True, exist_ok=True)

shutil.copytree(
    LOCAL_PROCESSED_22,
    DRIVE_PROCESSED_22,
    dirs_exist_ok=True
)

print("Copied Section 2.2 outputs to Drive:")
print(DRIVE_PROCESSED_22)

print("\nFiles inside parameters:")
for p in (DRIVE_PROCESSED_22 / "parameters").glob("*"):
    print(p.name)
```

## notebooks/2_3_rolling_baseline_no_enhancement.ipynb

### Cell 2: 20, OW
```python
!pip -q install pyarrow

import os
import gc
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

pd.set_option("display.max_columns", 180)
pd.set_option("display.width", 220)
```

### Cell 4: rolling, universe, missing, impact
```python
LOCAL_BASE = Path("/content/project_data")

DRIVE_PROJECT_DATA = Path(
    "/content/drive/MyDrive/Quantitative Trading and Price Impact/project_data"
)

DRIVE_PROCESSED_21 = DRIVE_PROJECT_DATA / "processed_2_1"
DRIVE_PROCESSED_22 = DRIVE_PROJECT_DATA / "processed_2_2_rolling_baseline"

SECTION21_DIR = LOCAL_BASE / "processed_2_1"
MONTHLY21_DIR = SECTION21_DIR / "monthly_standardised"
ROLLING21_DIR = SECTION21_DIR / "rolling_full_universe"

SECTION22_DIR = LOCAL_BASE / "processed_2_2_rolling_baseline"
PARAM22_DIR = SECTION22_DIR / "parameters"

SECTION23_DIR = LOCAL_BASE / "processed_2_3_rolling_baseline"
DAILY_DIR = SECTION23_DIR / "daily_summaries"
TRADE_LOG_DIR = SECTION23_DIR / "trade_logs"
REPORT_DIR = SECTION23_DIR / "report_tables"
FIG_DIR = SECTION23_DIR / "figures"
LOG_DIR = SECTION23_DIR / "logs"

# Restore 2.1 outputs
if not ROLLING21_DIR.exists():
    print("Restoring 2.1 outputs from Drive...")
    if not DRIVE_PROCESSED_21.exists():
        raise FileNotFoundError(f"Missing 2.1 Drive folder: {DRIVE_PROCESSED_21}")
    shutil.copytree(DRIVE_PROCESSED_21, SECTION21_DIR, dirs_exist_ok=True)

# Restore 2.2 outputs
if not PARAM22_DIR.exists():
    print("Restoring 2.2 outputs from Drive...")
    if not DRIVE_PROCESSED_22.exists():
        raise FileNotFoundError(f"Missing 2.2 Drive folder: {DRIVE_PROCESSED_22}")
    shutil.copytree(DRIVE_PROCESSED_22, SECTION22_DIR, dirs_exist_ok=True)

for p in [SECTION23_DIR, DAILY_DIR, TRADE_LOG_DIR, REPORT_DIR, FIG_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("2.1 rolling folder:", ROLLING21_DIR)
print("2.1 monthly folder:", MONTHLY21_DIR)
print("2.2 parameter folder:", PARAM22_DIR)
print("2.3 output folder:", SECTION23_DIR)

print("\n2.1 rolling files:")
for p in ROLLING21_DIR.glob("*"):
    print(p.name)

print("\n2.2 parameter files:")
for p in PARAM22_DIR.glob("*"):
    print(p.name)
```

### Cell 6: rolling, universe, 20, stock, missing, OW, reduced, train, test
```python
pair_summary_path = ROLLING21_DIR / "rolling_pair_summary.csv"
universe_path = ROLLING21_DIR / "rolling_universe_by_pair.csv"
ow_param_path = PARAM22_DIR / "ow_transient_params_by_pair_stock.csv"
rf_param_path = PARAM22_DIR / "reduced_form_params_by_pair_stock.csv"

required = [pair_summary_path, universe_path, ow_param_path, rf_param_path]
missing = [p for p in required if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Missing required files. Run enhanced 2.1 and then 2_2_rolling_baseline_no_enhancement.ipynb first. Missing:\n" +
        "\n".join(str(p) for p in missing)
    )

pair_summary = pd.read_csv(pair_summary_path)
pair_universe = pd.read_csv(universe_path)
ow_params = pd.read_csv(ow_param_path)
rf_params = pd.read_csv(rf_param_path)

for df in [pair_summary, pair_universe, ow_params, rf_params]:
    if "train_month" in df.columns:
        df["train_month"] = df["train_month"].astype(str)
    if "test_month" in df.columns:
        df["test_month"] = df["test_month"].astype(str)
    if "stock" in df.columns:
        df["stock"] = df["stock"].astype(str)

print("Rolling pairs:")
display(pair_summary[["pair_id", "train_month", "test_month"]])

print("Full universe size by pair:")
display(pair_universe.groupby("pair_id")["stock"].nunique().reset_index(name="n_full_universe_stocks"))

print("OW parameter rows:", ow_params.shape)
print("Reduced-form parameter rows:", rf_params.shape)
print("Confirmed: this notebook does not select top-20 stocks.")
```

### Cell 8: window, universe, 20, stock, volume, OW
```python
RUN_PAIR_IDS: Optional[List[int]] = None       # example for debugging: [1]
MAX_STOCKS_PER_PAIR: Optional[int] = None     # keep None for full universe; do not set to 20

# Generic baseline trade schedule.
PARTICIPATION_RATE = 0.005        # 0.5% of each stock-day public absolute volume
MIN_TARGET_SHARES = 1.0
OPEN_WINDOW_SECONDS = 30 * 60
CLOSE_WINDOW_SECONDS = 30 * 60
ROUND_LOT = 1.0

SAVE_NONZERO_TRADE_LOGS = True
SAVE_MAX_TRADE_LOG_ROWS_PER_PAIR = 300_000

BIN_COLUMNS_NEEDED = [
    "date", "time", "stock", "trade", "orderFlow", "hidden", "auction",
    "mid", "midEnd", "spread", "lobImb", "effLobImb", "depth",
    "datetime", "trading_date", "seconds_from_open"
]
```

### Cell 10: universe, stock, missing, fillna, OW, train
```python
def monthly_bin_path(month: str) -> Path:
    return MONTHLY21_DIR / "bin" / f"bin_{str(month)}_standardised.parquet"


def load_bin_month_for_universe(month: str, stocks: List[str], columns: Optional[List[str]] = None) -> pd.DataFrame:
    path = monthly_bin_path(month)
    if not path.exists():
        raise FileNotFoundError(f"Missing monthly bin Parquet for {month}: {path}")
    try:
        df = pd.read_parquet(path, columns=columns, filters=[("stock", "in", list(stocks))])
    except Exception:
        df = pd.read_parquet(path, columns=columns)
        df["stock"] = df["stock"].astype(str)
        df = df[df["stock"].isin(set(stocks))].copy()
    df["stock"] = df["stock"].astype(str)
    return df.reset_index(drop=True)


def get_pair_stock_universe(pair_id: int) -> List[str]:
    stocks = (
        pair_universe.loc[pair_universe["pair_id"] == pair_id, "stock"]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if MAX_STOCKS_PER_PAIR is not None:
        stocks = stocks[:MAX_STOCKS_PER_PAIR]
    return stocks


def make_seconds_from_open(df: pd.DataFrame) -> pd.Series:
    dt = pd.to_datetime(df["datetime"])
    open_time = pd.to_datetime(dt.dt.date.astype(str) + " 09:30:00")
    return (dt - open_time).dt.total_seconds().astype("float64")


def compute_training_scales(train_df: pd.DataFrame) -> Dict:
    tmp = train_df.copy()
    tmp["stock"] = tmp["stock"].astype(str)
    tmp["trading_date"] = tmp["trading_date"].astype(str)
    tmp["abs_orderFlow"] = pd.to_numeric(tmp["orderFlow"], errors="coerce").abs()

    daily_abs_flow = (
        tmp.groupby(["stock", "trading_date"], observed=True)["abs_orderFlow"]
        .sum()
        .reset_index(name="daily_abs_orderFlow")
    )
    stock_flow_scale = daily_abs_flow.groupby("stock", observed=True)["daily_abs_orderFlow"].median().replace(0, np.nan)
    global_flow_scale = float(np.nanmedian(stock_flow_scale.values))
    if not np.isfinite(global_flow_scale) or global_flow_scale <= 0:
        global_flow_scale = 1.0
    stock_flow_scale = stock_flow_scale.fillna(global_flow_scale).to_dict()

    median_depth = tmp.groupby("stock", observed=True)["depth"].median().replace(0, np.nan)
    global_depth_scale = float(np.nanmedian(median_depth.values))
    if not np.isfinite(global_depth_scale) or global_depth_scale <= 0:
        global_depth_scale = 1.0
    median_depth = median_depth.fillna(global_depth_scale).to_dict()

    return {
        "stock_flow_scale": {str(k): float(v) for k, v in stock_flow_scale.items()},
        "global_flow_scale": float(global_flow_scale),
        "stock_depth_scale": {str(k): float(v) for k, v in median_depth.items()},
        "global_depth_scale": float(global_depth_scale),
    }


def prepare_market_frame(df: pd.DataFrame, scales: Dict) -> pd.DataFrame:
    out = df.copy()
    out["stock"] = out["stock"].astype(str)
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["trading_date"] = out["trading_date"].astype(str)
    if "seconds_from_open" not in out.columns:
        out["seconds_from_open"] = make_seconds_from_open(out)

    out = out[out["datetime"].notna()].copy()
    out = out[(pd.to_numeric(out["mid"], errors="coerce") > 0)].copy()
    out = out.replace([np.inf, -np.inf], np.nan)

```

### Cell 12: window, stock, volume, OW
```python
def round_to_lot(x: float, lot: float = 1.0) -> float:
    if lot is None or lot <= 1:
        return float(x)
    return float(round(x / lot) * lot)


def choose_open_close_indices(day_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    seconds = pd.to_numeric(day_df["seconds_from_open"], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(seconds).sum() < 4:
        return np.array([], dtype=int), np.array([], dtype=int)

    max_sec = np.nanmax(seconds)
    open_pos = np.where(seconds <= OPEN_WINDOW_SECONDS)[0]
    close_pos = np.where(seconds >= max_sec - CLOSE_WINDOW_SECONDS)[0]

    if len(open_pos) == 0 or len(close_pos) == 0 or len(set(open_pos).intersection(set(close_pos))) > 0:
        n = len(day_df)
        k = max(1, int(0.2 * n))
        open_pos = np.arange(0, k)
        close_pos = np.arange(n - k, n)

    return open_pos.astype(int), close_pos.astype(int)


def add_daily_round_trip_trades(stock_df: pd.DataFrame) -> pd.DataFrame:
    out = stock_df.sort_values(["trading_date", "datetime"]).reset_index(drop=True).copy()
    out["our_trade"] = 0.0

    for _, idx in out.groupby("trading_date", observed=True, sort=False).indices.items():
        day = out.iloc[idx].copy()
        if len(day) < 4:
            continue

        abs_public_volume = float(pd.to_numeric(day["trade"], errors="coerce").abs().sum())
        if not np.isfinite(abs_public_volume) or abs_public_volume <= 0:
            abs_public_volume = float(pd.to_numeric(day["orderFlow"], errors="coerce").abs().sum())

        target_shares = round_to_lot(PARTICIPATION_RATE * abs_public_volume, ROUND_LOT)
        if not np.isfinite(target_shares) or target_shares < MIN_TARGET_SHARES:
            continue

        open_pos, close_pos = choose_open_close_indices(day)
        if len(open_pos) == 0 or len(close_pos) == 0:
            continue

        global_open_idx = np.asarray(idx)[open_pos]
        global_close_idx = np.asarray(idx)[close_pos]

        out.loc[global_open_idx, "our_trade"] += target_shares / len(global_open_idx)
        out.loc[global_close_idx, "our_trade"] -= target_shares / len(global_close_idx)

    return out
```

### Cell 14: stock, fillna, beta, OW, reduced, impact, impact_state, train, test
```python
def param_value(param_row: pd.Series, name: str, default: float = 0.0) -> float:
    if name not in param_row.index:
        return float(default)
    value = param_row[name]
    if pd.isna(value):
        return float(default)
    return float(value)


def daily_summary_from_path(path_df: pd.DataFrame, model: str) -> Dict:
    nonzero = path_df[path_df["our_trade"].abs() > 0].copy()
    if len(path_df) == 0:
        return {}

    first = path_df.iloc[0]
    last = path_df.iloc[-1]
    gross_shares = float(nonzero["our_trade"].abs().sum()) if len(nonzero) else 0.0
    net_shares = float(nonzero["our_trade"].sum()) if len(nonzero) else 0.0
    turnover_notional = float((nonzero["our_trade"].abs() * nonzero["mid"]).sum()) if len(nonzero) else 0.0

    return {
        "pair_id": int(first["pair_id"]),
        "train_month": str(first["train_month"]),
        "test_month": str(first["test_month"]),
        "model": model,
        "stock": str(first["stock"]),
        "trading_date": str(first["trading_date"]),
        "n_market_rows": int(len(path_df)),
        "n_our_trade_rows": int((path_df["our_trade"].abs() > 0).sum()),
        "gross_shares": gross_shares,
        "net_shares": net_shares,
        "turnover_notional": turnover_notional,
        "final_inventory": float(last["inventory"]),
        "final_cash": float(last["cash"]),
        "final_portfolio_value": float(last["portfolio_value"]),
        "execution_shortfall": float(path_df["execution_shortfall"].sum()),
        "spread_cost": float(path_df["spread_cost"].sum()),
        "impact_cost": float(path_df["impact_cost"].sum()),
        "max_abs_inventory": float(path_df["inventory"].abs().max()),
        "max_abs_impact_bps": float(path_df["impact_bps"].abs().max()),
        "max_abs_impact_state": float(path_df["impact_state_post"].abs().max()),
        "first_mid": float(first["mid"]),
        "last_mid": float(last["midEnd"]),
    }


def simulate_stock_day_ow(day_df: pd.DataFrame, param_row: pd.Series) -> Tuple[pd.DataFrame, Dict]:
    d = day_df.sort_values("datetime").reset_index(drop=True).copy()

    beta_flow = param_value(param_row, "x_flow", 0.0)
    beta_state = param_value(param_row, "ow_state_pre", 0.0)
    half_life_sec = param_value(param_row, "half_life_sec", 300.0)
    if not np.isfinite(half_life_sec) or half_life_sec <= 0:
        half_life_sec = 300.0
    tau = half_life_sec / math.log(2.0)

    n = len(d)
    state_pre_arr = np.zeros(n)
    state_post_arr = np.zeros(n)
    impact_bps_arr = np.zeros(n)
    exec_price_arr = np.full(n, np.nan)
    cash_arr = np.zeros(n)
    inventory_arr = np.zeros(n)
    shortfall_arr = np.zeros(n)
    spread_cost_arr = np.zeros(n)
    impact_cost_arr = np.zeros(n)
    portfolio_value_arr = np.zeros(n)

    state = 0.0
    inventory = 0.0
    cash = 0.0
    prev_sec = None

    seconds = pd.to_numeric(d["seconds_from_open"], errors="coerce").to_numpy(dtype=float)
    mids = pd.to_numeric(d["mid"], errors="coerce").to_numpy(dtype=float)
    mid_ends = pd.to_numeric(d["midEnd"], errors="coerce").fillna(d["mid"]).to_numpy(dtype=float)
    spreads_bps = pd.to_numeric(d["spread_bps"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    flow_scales = pd.to_numeric(d["flow_scale"], errors="coerce").replace(0, np.nan).fillna(1.0).to_numpy(dtype=float)
    trades = pd.to_numeric(d["our_trade"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

```

### Cell 16: universe, stock, dropna, OW, reduced, fit, train, test, backtest
```python
def get_param_row(params_df: pd.DataFrame, pair_id: int, stock: str) -> Optional[pd.Series]:
    m = params_df[(params_df["pair_id"] == pair_id) & (params_df["stock"].astype(str) == str(stock))]
    if len(m) == 0:
        return None
    return m.iloc[0]


def process_one_pair_backtest(pair_row: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pair_id = int(pair_row["pair_id"])
    train_month = str(pair_row["train_month"])
    test_month = str(pair_row["test_month"])

    stocks = get_pair_stock_universe(pair_id)
    print(f"\nPair {pair_id}: train {train_month} -> test {test_month}")
    print(f"Full common universe size: {len(stocks)}")

    train_raw = load_bin_month_for_universe(train_month, stocks, columns=BIN_COLUMNS_NEEDED)
    scales = compute_training_scales(train_raw)
    del train_raw
    gc.collect()

    test_raw = load_bin_month_for_universe(test_month, stocks, columns=BIN_COLUMNS_NEEDED)
    test = prepare_market_frame(test_raw, scales)
    del test_raw
    gc.collect()

    test["pair_id"] = pair_id
    test["train_month"] = train_month
    test["test_month"] = test_month

    stocks_with_data = sorted(test["stock"].dropna().astype(str).unique())
    ow_stocks = set(ow_params.loc[ow_params["pair_id"] == pair_id, "stock"].astype(str))
    rf_stocks = set(rf_params.loc[rf_params["pair_id"] == pair_id, "stock"].astype(str))
    stocks_to_simulate = [s for s in stocks_with_data if s in ow_stocks or s in rf_stocks]

    print("Stocks with test data:", len(stocks_with_data))
    print("Stocks with at least one fitted baseline model:", len(stocks_to_simulate))

    daily_rows = []
    failure_rows = []
    trade_log_parts = []

    stock_groups = {str(k): v.reset_index(drop=True) for k, v in test.groupby("stock", observed=True, sort=False)}

    for j, stock in enumerate(stocks_to_simulate, start=1):
        if j % 25 == 0 or j == 1 or j == len(stocks_to_simulate):
            print(f"  simulating stock {j}/{len(stocks_to_simulate)}: {stock}")

        stock_market = stock_groups.get(stock, pd.DataFrame())
        if len(stock_market) == 0:
            continue

        stock_with_trades = add_daily_round_trip_trades(stock_market)
        ow_row = get_param_row(ow_params, pair_id, stock)
        rf_row = get_param_row(rf_params, pair_id, stock)

        for trading_date, day_df in stock_with_trades.groupby("trading_date", observed=True, sort=False):
            day_df = day_df.reset_index(drop=True)
            if day_df["our_trade"].abs().sum() <= 0:
                continue

            if ow_row is not None:
                try:
                    path, summary = simulate_stock_day_ow(day_df, ow_row)
                    daily_rows.append(summary)
                    if SAVE_NONZERO_TRADE_LOGS:
                        trade_log_parts.append(path[path["our_trade"].abs() > 0].copy())
                except Exception as e:
                    failure_rows.append({"pair_id": pair_id, "stock": stock, "trading_date": str(trading_date), "model": "OW_transient", "reason": repr(e)})

            if rf_row is not None:
                try:
                    path, summary = simulate_stock_day_reduced_form(day_df, rf_row)
                    daily_rows.append(summary)
                    if SAVE_NONZERO_TRADE_LOGS:
                        trade_log_parts.append(path[path["our_trade"].abs() > 0].copy())
                except Exception as e:
                    failure_rows.append({"pair_id": pair_id, "stock": stock, "trading_date": str(trading_date), "model": "reduced_form", "reason": repr(e)})

    daily_df = pd.DataFrame(daily_rows)
```

### Cell 17: rolling, OW, train, test, backtest
```python
pairs_to_run = pair_summary.copy()
if RUN_PAIR_IDS is not None:
    pairs_to_run = pairs_to_run[pairs_to_run["pair_id"].isin(RUN_PAIR_IDS)].copy()

print("Pairs to backtest:")
display(pairs_to_run[["pair_id", "train_month", "test_month"]])

all_daily_parts = []
all_failure_parts = []

for _, pair_row in pairs_to_run.iterrows():
    daily_df, failures_df = process_one_pair_backtest(pair_row)
    all_daily_parts.append(daily_df)
    all_failure_parts.append(failures_df)

    combined_daily = pd.concat(all_daily_parts, ignore_index=True) if all_daily_parts else pd.DataFrame()
    combined_failures = pd.concat(all_failure_parts, ignore_index=True) if all_failure_parts else pd.DataFrame()

    combined_daily.to_csv(SECTION23_DIR / "section_2_3_rolling_baseline_daily_backtest_all_pairs.csv", index=False)
    combined_failures.to_csv(LOG_DIR / "section_2_3_rolling_baseline_failures_all_pairs.csv", index=False)

print("Section 2.3 rolling baseline backtest complete.")
```

### Cell 19: rolling, stock, dropna, OW, train, test, backtest
```python
daily_all_path = SECTION23_DIR / "section_2_3_rolling_baseline_daily_backtest_all_pairs.csv"
daily_all = pd.read_csv(daily_all_path)

print("Daily backtest rows:", daily_all.shape)
display(daily_all.head())

print("Models simulated:")
display(daily_all["model"].value_counts(dropna=False))

print("Pairs simulated:")
display(daily_all.groupby(["pair_id", "train_month", "test_month"])["stock"].nunique().reset_index(name="n_stocks_simulated"))
```

### Cell 20: stock, impact
```python
summary_by_model = (
    daily_all.groupby("model", observed=True)
    .agg(
        n_pair_days=("trading_date", "count"),
        n_pairs=("pair_id", "nunique"),
        n_stocks=("stock", "nunique"),
        total_turnover_notional=("turnover_notional", "sum"),
        total_execution_shortfall=("execution_shortfall", "sum"),
        total_spread_cost=("spread_cost", "sum"),
        total_impact_cost=("impact_cost", "sum"),
        avg_daily_shortfall=("execution_shortfall", "mean"),
        avg_daily_spread_cost=("spread_cost", "mean"),
        avg_daily_impact_cost=("impact_cost", "mean"),
        avg_max_abs_inventory=("max_abs_inventory", "mean"),
        avg_max_abs_impact_bps=("max_abs_impact_bps", "mean"),
    )
    .reset_index()
)

summary_by_model["shortfall_bps_of_turnover"] = 10000.0 * summary_by_model["total_execution_shortfall"] / summary_by_model["total_turnover_notional"].replace(0, np.nan)
summary_by_model["spread_cost_bps_of_turnover"] = 10000.0 * summary_by_model["total_spread_cost"] / summary_by_model["total_turnover_notional"].replace(0, np.nan)
summary_by_model["impact_cost_bps_of_turnover"] = 10000.0 * summary_by_model["total_impact_cost"] / summary_by_model["total_turnover_notional"].replace(0, np.nan)
summary_by_model.to_csv(REPORT_DIR / "section_2_3_summary_by_model.csv", index=False)
display(summary_by_model)
```

### Cell 21: 20, stock, impact, train, test
```python
summary_by_pair_model = (
    daily_all.groupby(["pair_id", "train_month", "test_month", "model"], observed=True)
    .agg(
        n_pair_days=("trading_date", "count"),
        n_stocks=("stock", "nunique"),
        total_turnover_notional=("turnover_notional", "sum"),
        total_execution_shortfall=("execution_shortfall", "sum"),
        total_spread_cost=("spread_cost", "sum"),
        total_impact_cost=("impact_cost", "sum"),
        avg_max_abs_impact_bps=("max_abs_impact_bps", "mean"),
    )
    .reset_index()
)
summary_by_pair_model["shortfall_bps_of_turnover"] = 10000.0 * summary_by_pair_model["total_execution_shortfall"] / summary_by_pair_model["total_turnover_notional"].replace(0, np.nan)
summary_by_pair_model["impact_cost_bps_of_turnover"] = 10000.0 * summary_by_pair_model["total_impact_cost"] / summary_by_pair_model["total_turnover_notional"].replace(0, np.nan)
summary_by_pair_model.to_csv(REPORT_DIR / "section_2_3_summary_by_pair_model.csv", index=False)
display(summary_by_pair_model.head(20))
```

### Cell 22: 20, stock, impact
```python
summary_by_stock_model = (
    daily_all.groupby(["model", "stock"], observed=True)
    .agg(
        n_days=("trading_date", "count"),
        total_turnover_notional=("turnover_notional", "sum"),
        total_execution_shortfall=("execution_shortfall", "sum"),
        total_impact_cost=("impact_cost", "sum"),
        avg_max_abs_impact_bps=("max_abs_impact_bps", "mean"),
    )
    .reset_index()
)
summary_by_stock_model["shortfall_bps_of_turnover"] = 10000.0 * summary_by_stock_model["total_execution_shortfall"] / summary_by_stock_model["total_turnover_notional"].replace(0, np.nan)
summary_by_stock_model = summary_by_stock_model.sort_values("shortfall_bps_of_turnover", ascending=False)
summary_by_stock_model.to_csv(REPORT_DIR / "section_2_3_summary_by_stock_model.csv", index=False)
display(summary_by_stock_model.head(20))
```

### Cell 24: OW, impact, fit
```python
plot_df = summary_by_model.set_index("model")["shortfall_bps_of_turnover"].sort_index()
ax = plot_df.plot(kind="bar", figsize=(8, 5))
ax.set_title("Section 2.3 baseline: execution shortfall by fitted impact model")
ax.set_xlabel("Fitted impact model")
ax.set_ylabel("Execution shortfall, bps of turnover")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "shortfall_bps_by_model.png", dpi=150)
plt.show()
```

### Cell 25: rolling, pivot, OW
```python
pair_pivot = summary_by_pair_model.pivot(index="pair_id", columns="model", values="shortfall_bps_of_turnover").sort_index()
ax = pair_pivot.plot(kind="bar", figsize=(12, 5))
ax.set_title("Section 2.3 baseline: shortfall by rolling pair")
ax.set_xlabel("Rolling pair id")
ax.set_ylabel("Execution shortfall, bps of turnover")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "shortfall_bps_by_pair_model.png", dpi=150)
plt.show()
```

### Cell 26: rolling, pivot, OW, impact
```python
impact_pivot = summary_by_pair_model.pivot(index="pair_id", columns="model", values="impact_cost_bps_of_turnover").sort_index()
ax = impact_pivot.plot(kind="bar", figsize=(12, 5))
ax.set_title("Section 2.3 baseline: estimated impact cost by rolling pair")
ax.set_xlabel("Rolling pair id")
ax.set_ylabel("Impact cost, bps of turnover")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "impact_cost_bps_by_pair_model.png", dpi=150)
plt.show()
```

### Cell 28: out-of-sample, rolling, window, universe, 20, stock, OW, reduced, impact, test, backtest
```python
handoff = {
    "section": "2.3 Backtest Engine rolling-universe baseline",
    "continues_from": "2.1 rolling_full_universe enhancement",
    "uses_20_stock_baseline": False,
    "uses_full_rolling_universe": True,
    "section_2_2_enhancement_used": False,
    "section_2_3_enhancement_used": False,
    "models_used_from_section_2_2": ["OW_transient", "reduced_form"],
    "backtest_sample": "out-of-sample test month for each rolling pair",
    "generic_trade_column": "our_trade",
    "baseline_daily_reset": True,
    "no_enhancement_details": [
        "No non-parametric or regularised model from Section 2.2.",
        "No multi-day carrying of impact states.",
        "No multi-day carrying of positions.",
        "No stock-split detection or exclusion logic.",
        "No alpha signal and no optimal strategy in this section.",
    ],
    "example_trade_schedule": {
        "type": "daily_round_trip",
        "participation_rate": PARTICIPATION_RATE,
        "open_window_seconds": OPEN_WINDOW_SECONDS,
        "close_window_seconds": CLOSE_WINDOW_SECONDS,
        "purpose": "baseline simulator demonstration only; not an optimal strategy",
    },
    "output_files": {
        "daily_backtest_all_pairs": str(SECTION23_DIR / "section_2_3_rolling_baseline_daily_backtest_all_pairs.csv"),
        "summary_by_model": str(REPORT_DIR / "section_2_3_summary_by_model.csv"),
        "summary_by_pair_model": str(REPORT_DIR / "section_2_3_summary_by_pair_model.csv"),
        "summary_by_stock_model": str(REPORT_DIR / "section_2_3_summary_by_stock_model.csv"),
    },
}

with open(SECTION23_DIR / "section_2_3_rolling_baseline_handoff.json", "w") as f:
    json.dump(handoff, f, indent=2)

print(json.dumps(handoff, indent=2))
```
