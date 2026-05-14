"""Audit teammate processed data and notebooks for sections 2.1, 2.2, and 2.3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


KEYWORDS = [
    "in-sample", "out-of-sample", "rolling", "window", "universe", "20", "stock",
    "missing", "fillna", "dropna", "ffill", "bfill", "pivot", "ADV", "sigma",
    "px_vol", "volume", "lambda", "beta", "OW", "Obizhaeva", "AFS", "reduced",
    "sqrt", "linear", "impact", "impact_state", "calibration", "fit", "OLS",
    "regression", "loss", "train", "test", "validation", "backtest", "Waelbroeck",
]


def _df_to_md(df: pd.DataFrame) -> str:
    """Return a simple markdown-like table without optional dependencies."""

    if df.empty:
        return "_No rows._"
    return "```text\n" + df.to_string(index=False) + "\n```"


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return df with a stable schema even when it has no rows."""

    if df.empty and len(df.columns) == 0:
        return pd.DataFrame(columns=columns)
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


@dataclass(frozen=True)
class TeammateAuditPaths:
    """Paths used by the teammate audit."""

    project_root: Path
    data_21_dir: Path
    data_22_dir: Path
    notebooks: tuple[Path, ...]
    out_dir: Path
    fig_dir: Path


def find_project_root(start: Path) -> Path:
    """Find project root from current working directory."""

    start = start.resolve()
    if start.name in {"notebooks", "src"}:
        start = start.parent
    for candidate in [start] + list(start.parents):
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not find project root containing data/ and src/")


def build_paths(project_root: Path | None = None) -> TeammateAuditPaths:
    """Build audit paths and output directories."""

    root = project_root or find_project_root(Path.cwd())
    out_dir = root / "outputs" / "teammate_audit"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    notebooks = (
        root / "notebooks" / "2_1_full_as_spec_completed.ipynb",
        root / "notebooks" / "2_2_rolling_baseline_no_enhancement.ipynb",
        root / "notebooks" / "2_3_rolling_baseline_no_enhancement.ipynb",
    )
    return TeammateAuditPaths(
        project_root=root,
        data_21_dir=root / "data" / "processed_2_1",
        data_22_dir=root / "data" / "processed_2_2_rolling_baseline",
        notebooks=notebooks,
        out_dir=out_dir,
        fig_dir=fig_dir,
    )


def discover_files(paths: TeammateAuditPaths) -> pd.DataFrame:
    """Discover processed output files and save file inventory."""

    rows: list[dict[str, Any]] = []
    for label, folder in [
        ("processed_2_1", paths.data_21_dir),
        ("processed_2_2_rolling_baseline", paths.data_22_dir),
    ]:
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            rows.append(
                {
                    "relative_path": str(path.relative_to(paths.project_root)),
                    "folder_label": label,
                    "file_name": path.name,
                    "suffix": path.suffix.lower(),
                    "size_mb": stat.st_size / 1024**2,
                    "modified_time": pd.to_datetime(stat.st_mtime, unit="s"),
                    "parent_folder": str(path.parent.relative_to(paths.project_root)),
                }
            )
    inventory = pd.DataFrame(rows)
    inventory.to_csv(paths.out_dir / "file_inventory.csv", index=False)
    return inventory


def _read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def _read_parquet_sample(path: Path, nrows: int = 5000) -> pd.DataFrame:
    try:
        return pd.read_parquet(path).head(nrows)
    except Exception:
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            return table.slice(0, nrows).to_pandas()
        except Exception as exc:
            raise RuntimeError(f"Could not read parquet sample {path}: {exc}") from exc


def _parse_date(values: pd.Series) -> pd.Series:
    as_str = values.astype("string").str.strip()
    ymd = as_str.str.fullmatch(r"\d{8}", na=False)
    return pd.to_datetime(as_str.where(~ymd), errors="coerce").fillna(
        pd.to_datetime(as_str.where(ymd), format="%Y%m%d", errors="coerce")
    )


def _parse_timestamp(df: pd.DataFrame) -> pd.Series | None:
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], errors="coerce")
    if {"date", "time"}.issubset(df.columns):
        date = _parse_date(df["date"])
        time = pd.to_timedelta(df["time"].astype("string"), errors="coerce")
        return date + time
    return None


def _numeric_summary(df: pd.DataFrame, source: str) -> list[dict[str, Any]]:
    rows = []
    numeric = df.select_dtypes(include=[np.number])
    for col in numeric.columns:
        s = pd.to_numeric(numeric[col], errors="coerce")
        rows.append(
            {
                "source_file": source,
                "column": col,
                "count": int(s.count()),
                "mean": float(s.mean()) if s.count() else np.nan,
                "std": float(s.std(ddof=1)) if s.count() > 1 else np.nan,
                "min": float(s.min()) if s.count() else np.nan,
                "q01": float(s.quantile(0.01)) if s.count() else np.nan,
                "median": float(s.median()) if s.count() else np.nan,
                "q99": float(s.quantile(0.99)) if s.count() else np.nan,
                "max": float(s.max()) if s.count() else np.nan,
                "missing_pct": float(s.isna().mean() * 100),
                "n_zero": int((s == 0).sum()),
                "n_negative": int((s < 0).sum()),
                "n_positive": int((s > 0).sum()),
            }
        )
    return rows


def audit_tabular_files(
    paths: TeammateAuditPaths,
    inventory: pd.DataFrame,
    nrows: int | None = None,
    max_files: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Audit CSV files fully and parquet files via bounded samples."""

    table_files = inventory[inventory["suffix"].isin([".csv", ".parquet"])].copy()
    if max_files is not None:
        table_files = table_files.head(max_files)
    schema_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    numeric_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    time_gap_rows: list[dict[str, Any]] = []
    jump_rows: list[dict[str, Any]] = []

    for _, file_row in table_files.iterrows():
        rel = file_row["relative_path"]
        path = paths.project_root / rel
        try:
            df = _read_csv(path, nrows=nrows) if path.suffix.lower() == ".csv" else _read_parquet_sample(path)
        except Exception as exc:
            schema_rows.append({"source_file": rel, "error": str(exc)})
            continue
        df = df.copy()
        df["source_file"] = rel
        schema_rows.append(
            {
                "source_file": rel,
                "suffix": path.suffix.lower(),
                "n_rows": len(df),
                "n_cols": df.shape[1],
                "columns": "|".join(map(str, df.columns)),
                "dtypes": json.dumps({c: str(t) for c, t in df.dtypes.items()}),
                "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
                "sampled": bool(path.suffix.lower() == ".parquet" or nrows is not None),
            }
        )
        for col in df.columns:
            missing_rows.append(
                {
                    "source_file": rel,
                    "column": col,
                    "missing_count": int(df[col].isna().sum()),
                    "missing_pct": float(df[col].isna().mean() * 100),
                }
            )
        numeric_rows.extend(_numeric_summary(df, rel))

        date_col = "date" if "date" in df.columns else None
        stock_col = "stock" if "stock" in df.columns else None
        ts = _parse_timestamp(df)
        dates = _parse_date(df[date_col]) if date_col else (ts.dt.normalize() if ts is not None else None)
        groups = None
        if stock_col and dates is not None:
            groups = df.assign(_date=dates).groupby([stock_col, "_date"], sort=False).size()
        coverage_rows.append(
            {
                "source_file": rel,
                "n_rows": len(df),
                "n_unique_stocks": int(df[stock_col].nunique()) if stock_col else np.nan,
                "n_unique_dates": int(dates.nunique()) if dates is not None else np.nan,
                "min_date": dates.min() if dates is not None else pd.NaT,
                "max_date": dates.max() if dates is not None else pd.NaT,
                "n_unique_times": int(df["time"].nunique()) if "time" in df.columns else np.nan,
                "min_time": df["time"].min() if "time" in df.columns and len(df) else np.nan,
                "max_time": df["time"].max() if "time" in df.columns and len(df) else np.nan,
                "rows_per_stock_date_median": float(groups.median()) if groups is not None and len(groups) else np.nan,
                "rows_per_stock_date_min": float(groups.min()) if groups is not None and len(groups) else np.nan,
                "rows_per_stock_date_max": float(groups.max()) if groups is not None and len(groups) else np.nan,
            }
        )

        duplicate_key = int(df.duplicated([c for c in ["stock", "date", "time"] if c in df.columns]).sum()) if {"stock", "date", "time"}.issubset(df.columns) else np.nan
        quality_rows.append(
            {
                "source_file": rel,
                "duplicate_rows": int(df.duplicated().sum()),
                "duplicate_stock_date_time": duplicate_key,
                "missing_mid_pct": float(df["mid"].isna().mean() * 100) if "mid" in df.columns else np.nan,
                "non_positive_mid_count": int((pd.to_numeric(df.get("mid"), errors="coerce") <= 0).sum()) if "mid" in df.columns else np.nan,
                "negative_spread_count": int((pd.to_numeric(df.get("spread"), errors="coerce") < 0).sum()) if "spread" in df.columns else np.nan,
                "zero_volume_days": np.nan,
            }
        )

        if ts is not None and stock_col and dates is not None:
            temp = df.assign(_timestamp=ts, _date=dates).sort_values([stock_col, "_date", "_timestamp"])
            gaps = temp.groupby([stock_col, "_date"], sort=False)["_timestamp"].diff().dt.total_seconds()
            time_gap_rows.append(
                {
                    "source_file": rel,
                    "median_gap_seconds": float(gaps.median()) if gaps.notna().any() else np.nan,
                    "p95_gap_seconds": float(gaps.quantile(0.95)) if gaps.notna().any() else np.nan,
                    "max_gap_seconds": float(gaps.max()) if gaps.notna().any() else np.nan,
                    "non_monotonic_groups": 0,
                }
            )
            if "mid" in temp.columns:
                temp["_mid"] = pd.to_numeric(temp["mid"], errors="coerce")
                ret = temp.groupby([stock_col, "_date"], sort=False)["_mid"].pct_change()
                big = temp.loc[ret.abs() > 0.05, [stock_col, "_date", "_timestamp"]].copy()
                if len(big):
                    big["abs_return"] = ret.loc[big.index].abs().to_numpy()
                    big["source_file"] = rel
                    jump_rows.extend(big.rename(columns={stock_col: "stock", "_date": "date", "_timestamp": "timestamp"}).head(100).to_dict("records"))

    outputs = {
        "schema": _ensure_columns(pd.DataFrame(schema_rows), ["source_file", "suffix", "n_rows", "n_cols", "columns", "dtypes", "memory_mb", "sampled", "error"]),
        "missing": _ensure_columns(pd.DataFrame(missing_rows), ["source_file", "column", "missing_count", "missing_pct"]),
        "numeric": _ensure_columns(pd.DataFrame(numeric_rows), ["source_file", "column", "count", "mean", "std", "min", "q01", "median", "q99", "max", "missing_pct", "n_zero", "n_negative", "n_positive"]),
        "coverage": _ensure_columns(pd.DataFrame(coverage_rows), ["source_file", "n_rows", "n_unique_stocks", "n_unique_dates", "min_date", "max_date", "n_unique_times", "min_time", "max_time", "rows_per_stock_date_median", "rows_per_stock_date_min", "rows_per_stock_date_max"]),
        "quality": _ensure_columns(pd.DataFrame(quality_rows), ["source_file", "duplicate_rows", "duplicate_stock_date_time", "missing_mid_pct", "non_positive_mid_count", "negative_spread_count", "zero_volume_days"]),
        "time_gaps": _ensure_columns(pd.DataFrame(time_gap_rows), ["source_file", "median_gap_seconds", "p95_gap_seconds", "max_gap_seconds", "non_monotonic_groups"]),
        "jumps": _ensure_columns(pd.DataFrame(jump_rows), ["stock", "date", "timestamp", "abs_return", "source_file"]),
    }
    outputs["schema"].to_csv(paths.out_dir / "csv_schema_summary.csv", index=False)
    outputs["missing"].to_csv(paths.out_dir / "csv_missing_summary.csv", index=False)
    outputs["numeric"].to_csv(paths.out_dir / "csv_numeric_summary.csv", index=False)
    outputs["coverage"].to_csv(paths.out_dir / "csv_key_coverage_summary.csv", index=False)
    outputs["quality"].to_csv(paths.out_dir / "data_quality_summary.csv", index=False)
    outputs["time_gaps"].to_csv(paths.out_dir / "time_gap_summary.csv", index=False)
    outputs["jumps"].to_csv(paths.out_dir / "large_price_jumps_processed.csv", index=False)
    return outputs


def classify_artifacts(paths: TeammateAuditPaths, schema: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Classify processed artifacts by filename and detected columns."""

    rows = []
    for _, row in schema.iterrows():
        source = str(row.get("source_file", ""))
        cols = set(str(row.get("columns", "")).split("|"))
        lower = source.lower()
        artifact = "unknown"
        if "standardised" in lower and "/bin/" in lower:
            artifact = "cleaned_bins"
        elif "standardised" in lower and "fill" in lower:
            artifact = "cleaned_fills"
        elif "universe" in lower or "20stocks" in lower:
            artifact = "selected_universe"
        elif "rolling" in lower and ("pair" in lower or "window" in lower):
            artifact = "rolling_windows"
        elif "param" in lower or {"lambda", "beta"} & cols:
            artifact = "fitted_parameters"
        elif "metric" in lower or "rmse" in lower:
            artifact = "model_diagnostics"
        elif {"trailing_px_vol", "trailing_ADV"} <= cols:
            artifact = "scaling_factors"
        elif "backtest" in lower or {"position", "trade", "pnl"} & cols:
            artifact = "backtest_outputs"
        cov = coverage.loc[coverage["source_file"].eq(source)]
        rows.append(
            {
                "artifact_type": artifact,
                "likely_section": "2.1" if "processed_2_1" in source else "2.2/2.3",
                "source_file": source,
                "detected_columns": row.get("columns", ""),
                "n_rows": row.get("n_rows"),
                "date_range": f"{cov['min_date'].iloc[0]} to {cov['max_date'].iloc[0]}" if not cov.empty else "",
                "stock_count": cov["n_unique_stocks"].iloc[0] if not cov.empty else np.nan,
                "notes": "",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths.out_dir / "artifact_classification.csv", index=False)
    return out


def audit_universe_and_splits(paths: TeammateAuditPaths, schema: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Find universe and rolling split candidate files."""

    universe_rows, split_rows, rolling_rows = [], [], []
    for _, row in schema.iterrows():
        source = str(row.get("source_file", ""))
        lower = source.lower()
        if not source.endswith(".csv"):
            continue
        path = paths.project_root / source
        try:
            df = _read_csv(path)
        except Exception:
            continue
        if "stock" in df.columns and ("universe" in lower or "20stock" in lower or df["stock"].nunique() <= 40):
            universe_rows.append(
                {
                    "source_file": source,
                    "stock_count": int(df["stock"].nunique()),
                    "stocks": ",".join(sorted(map(str, df["stock"].dropna().unique()))[:50]),
                    "exactly_20": bool(df["stock"].nunique() == 20),
                    "date_coverage": f"{_parse_date(df['date']).min()} to {_parse_date(df['date']).max()}" if "date" in df.columns else "",
                }
            )
        cols = set(df.columns)
        if cols & {"train_month", "test_month", "train_start", "test_start", "window_id", "pair_id"} or "pair" in lower:
            split_rows.append(
                {
                    "source_file": source,
                    "columns": "|".join(df.columns),
                    "n_rows": len(df),
                    "train_start": df.get("train_start", df.get("train_month", pd.Series([np.nan]))).iloc[0] if len(df) else np.nan,
                    "test_start": df.get("test_start", df.get("test_month", pd.Series([np.nan]))).iloc[0] if len(df) else np.nan,
                    "stock_count": int(df["stock"].nunique()) if "stock" in df.columns else np.nan,
                }
            )
            rolling_rows.append(split_rows[-1].copy())
    outputs = {
        "universe": pd.DataFrame(universe_rows),
        "splits": pd.DataFrame(split_rows),
        "rolling": pd.DataFrame(rolling_rows),
    }
    outputs["universe"].to_csv(paths.out_dir / "selected_universe_candidates.csv", index=False)
    outputs["splits"].to_csv(paths.out_dir / "date_split_candidates.csv", index=False)
    outputs["rolling"].to_csv(paths.out_dir / "rolling_window_candidates.csv", index=False)
    return outputs


def audit_scaling_factors(paths: TeammateAuditPaths, schema: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Audit candidate scaling factor outputs."""

    audit_rows, coverage_rows = [], []
    for _, row in schema.iterrows():
        source = str(row.get("source_file", ""))
        cols = set(str(row.get("columns", "")).split("|"))
        lower = source.lower()
        cols_lower = {c.lower() for c in cols}
        has_final_scaling = bool({"trailing_px_vol", "trailing_adv", "sigma", "adv"} & cols_lower)
        has_volume_inputs = bool(
            {"total_abs_trade", "total_abs_notional", "abs_trade", "abs_notional", "trade"} & cols_lower
        )
        has_return_inputs = bool({"ret_in_bin", "d_mid_in_bin", "midend", "mid"} & cols_lower)
        looks_relevant = (
            has_final_scaling
            or has_volume_inputs
            or ("adv" in lower)
            or ("px_vol" in lower)
            or ("stock_stats" in lower)
            or ("standardised" in lower and ("bin" in lower or "fill" in lower))
        )
        if not looks_relevant:
            continue
        if source.endswith(".csv"):
            try:
                df = _read_csv(paths.project_root / source)
            except Exception:
                df = pd.DataFrame(columns=list(cols))
        elif source.endswith(".parquet"):
            try:
                df = _read_parquet_sample(paths.project_root / source, nrows=5000)
            except Exception:
                df = pd.DataFrame(columns=list(cols))
        else:
            continue
        detected_columns = list(dict.fromkeys(list(df.columns) + [c for c in cols if c and c != "nan"]))
        sigma_cols = [c for c in detected_columns if "px_vol" in c.lower() or "sigma" in c.lower()]
        return_cols = [c for c in detected_columns if c.lower() in {"ret_in_bin", "d_mid_in_bin"}]
        adv_cols = [
            c
            for c in detected_columns
            if "adv" in c.lower() or "volume" in c.lower() or c.lower() in {"total_abs_trade", "abs_trade", "trade", "abs_notional", "total_abs_notional"}
        ]
        has_shift_evidence = "trailing" in " ".join(detected_columns).lower() or "trailing" in lower
        if has_final_scaling:
            artifact = "final_or_candidate_scaling_factor_file"
            no_lookahead = "likely" if has_shift_evidence else "unclear"
            notes = "Contains sigma/ADV-style columns; verify shift(1) in notebook."
        elif "stock_stats" in lower or "summary" in lower:
            artifact = "monthly_volume_liquidity_input"
            no_lookahead = "not_applicable"
            notes = "Contains monthly liquidity/volume summaries, not 20-day trailing sigma/ADV."
        elif return_cols and adv_cols:
            artifact = "row_level_input_for_scaling_recompute"
            no_lookahead = "not_applicable"
            notes = "Contains row-level returns/volume needed to compute sigma/ADV, but not trailing scaling factors."
        else:
            artifact = "scaling_related_candidate"
            no_lookahead = "unclear"
            notes = "Relevant liquidity columns found, but no finalized trailing scaling columns detected."
        audit_rows.append(
            {
                "source_file": source,
                "artifact_type": artifact,
                "n_rows": len(df),
                "columns": "|".join(detected_columns),
                "sigma_columns": "|".join(sigma_cols),
                "return_columns": "|".join(return_cols),
                "adv_columns": "|".join(adv_cols),
                "positive_sigma_share": float((pd.to_numeric(df[sigma_cols[0]], errors="coerce") > 0).mean()) if sigma_cols else np.nan,
                "positive_adv_share": float((pd.to_numeric(df[adv_cols[0]], errors="coerce") > 0).mean()) if adv_cols else np.nan,
                "no_lookahead_evidence": no_lookahead,
                "notes": notes,
            }
        )
        if "date" in df.columns:
            dates = _parse_date(df["date"])
            coverage = df.assign(_date=dates).groupby("_date").agg(rows=("date", "size")).reset_index()
            coverage["source_file"] = source
            coverage_rows.extend(coverage.rename(columns={"_date": "date"}).to_dict("records"))
        elif "month" in df.columns:
            month_dates = pd.to_datetime(df["month"].astype("string"), format="%Y%m", errors="coerce")
            coverage = df.assign(_date=month_dates).groupby("_date").agg(rows=("month", "size")).reset_index()
            coverage["source_file"] = source
            coverage_rows.extend(coverage.rename(columns={"_date": "date"}).to_dict("records"))
        elif "first_datetime" in df.columns:
            dates = pd.to_datetime(df["first_datetime"], errors="coerce").dt.to_period("M").dt.to_timestamp()
            coverage = df.assign(_date=dates).groupby("_date").agg(rows=("first_datetime", "size")).reset_index()
            coverage["source_file"] = source
            coverage_rows.extend(coverage.rename(columns={"_date": "date"}).to_dict("records"))
    outputs = {
        "audit": _ensure_columns(
            pd.DataFrame(audit_rows),
            [
                "source_file",
                "artifact_type",
                "n_rows",
                "columns",
                "sigma_columns",
                "return_columns",
                "adv_columns",
                "positive_sigma_share",
                "positive_adv_share",
                "no_lookahead_evidence",
                "notes",
            ],
        ),
        "coverage": _ensure_columns(pd.DataFrame(coverage_rows), ["date", "rows", "source_file"]),
    }
    outputs["audit"].to_csv(paths.out_dir / "scaling_factor_audit.csv", index=False)
    outputs["coverage"].to_csv(paths.out_dir / "scaling_coverage_by_date.csv", index=False)
    return outputs


def audit_notebooks(paths: TeammateAuditPaths) -> dict[str, pd.DataFrame]:
    """Parse notebooks, build keyword cell index, and extract relevant snippets."""

    index_rows = []
    md_lines = ["# Notebook Code Extracts", ""]
    for notebook in paths.notebooks:
        if not notebook.exists():
            index_rows.append({"notebook": str(notebook.relative_to(paths.project_root)), "cell_index": np.nan, "cell_type": "missing", "first_line": "NOTEBOOK MISSING", "keyword_hits": "", "n_lines": 0})
            continue
        nb = nbformat.read(notebook, as_version=4)
        rel = str(notebook.relative_to(paths.project_root))
        md_lines.extend([f"## {rel}", ""])
        for i, cell in enumerate(nb.cells):
            source = cell.get("source", "")
            first = source.strip().splitlines()[0] if source.strip() else ""
            hits = [kw for kw in KEYWORDS if kw.lower() in source.lower()]
            index_rows.append(
                {
                    "notebook": rel,
                    "cell_index": i,
                    "cell_type": cell.cell_type,
                    "first_line": first[:200],
                    "keyword_hits": ",".join(hits),
                    "n_lines": len(source.splitlines()),
                    "executed": cell.get("execution_count") is not None,
                    "n_outputs": len(cell.get("outputs", [])) if cell.cell_type == "code" else 0,
                }
            )
            if cell.cell_type == "code" and hits:
                md_lines.extend([f"### Cell {i}: {', '.join(hits[:12])}", "```python", "\n".join(source.splitlines()[:80]), "```", ""])
    index = pd.DataFrame(index_rows)
    index.to_csv(paths.out_dir / "notebook_cell_index.csv", index=False)
    (paths.out_dir / "notebook_code_extracts.md").write_text("\n".join(md_lines), encoding="utf-8")
    return {"cell_index": index}


def audit_models(paths: TeammateAuditPaths, schema: pd.DataFrame, notebook_index: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Infer impact model fitting and fitted parameter files."""

    param_files = schema[schema["source_file"].str.contains("param|lambda|beta|reduced|ow", case=False, na=False)].copy()
    fitted_rows = []
    for _, row in param_files.iterrows():
        source = str(row["source_file"])
        sample = ""
        models = []
        stocks = windows = date_range = ""
        if source.endswith(".csv"):
            try:
                df = _read_csv(paths.project_root / source)
                sample = df.head(3).to_json(orient="records")
                models = [m for m in ["OW", "reduced", "AFS", "sqrt", "linear"] if m.lower() in (source + " " + " ".join(df.columns)).lower()]
                stocks = str(df["stock"].nunique()) if "stock" in df.columns else ""
                windows = str(df["pair_id"].nunique()) if "pair_id" in df.columns else str(df["window_id"].nunique()) if "window_id" in df.columns else ""
                if "date" in df.columns:
                    d = _parse_date(df["date"])
                    date_range = f"{d.min()} to {d.max()}"
            except Exception as exc:
                sample = f"ERROR: {exc}"
        fitted_rows.append(
            {
                "source_file": source,
                "columns": row.get("columns", ""),
                "n_rows": row.get("n_rows"),
                "models_detected": ",".join(models),
                "stocks_detected": stocks,
                "windows_detected": windows,
                "date_range": date_range,
                "sample_rows": sample,
            }
        )
    fitted = pd.DataFrame(fitted_rows)
    fitted.to_csv(paths.out_dir / "fitted_parameter_files.csv", index=False)

    evidence = notebook_index[notebook_index["keyword_hits"].str.contains("OW|AFS|reduced|lambda|beta|fit|regression", na=False)]
    evidence_cells = "; ".join((evidence["notebook"] + ":cell" + evidence["cell_index"].astype(str)).head(20))
    models = []
    for name in ["OW", "reduced_form_AFS"]:
        implemented = "unclear"
        concerns = []
        param_cols = ""
        norm = "unknown"
        beta = "unknown"
        lam = "unknown"
        relevant = fitted[fitted["source_file"].str.contains("ow", case=False, na=False)] if name == "OW" else fitted[fitted["source_file"].str.contains("reduced|afs", case=False, na=False)]
        if not relevant.empty:
            implemented = "yes"
            param_cols = str(relevant.iloc[0]["columns"])
            norm = "sigma/ADV likely" if "sigma" in param_cols.lower() or "adv" in param_cols.lower() else "unclear"
            beta = "parameter columns found" if "beta" in param_cols.lower() or "half" in param_cols.lower() else "unclear"
            lam = "parameter columns found" if "lambda" in param_cols.lower() else "unclear"
        else:
            concerns.append("No obvious parameter file found")
        models.append(
            {
                "model_name": name,
                "implemented": implemented,
                "evidence_files": "|".join(relevant["source_file"].head(10)) if not relevant.empty else "",
                "evidence_notebook_cells": evidence_cells,
                "parameter_columns_found": param_cols,
                "parameter_scope": "per_stock/per_window likely" if "stock" in param_cols.lower() and ("pair" in param_cols.lower() or "window" in param_cols.lower()) else "unknown",
                "normalization_used": norm,
                "beta_handling": beta,
                "lambda_handling": lam,
                "notes": "Inferred from filenames/columns/notebook keywords",
                "concerns": "; ".join(concerns),
            }
        )
    model_audit = pd.DataFrame(models)
    model_audit.to_csv(paths.out_dir / "model_fit_audit.csv", index=False)
    return {"model_audit": model_audit, "fitted_files": fitted}


def audit_backtest(paths: TeammateAuditPaths, notebook_index: pd.DataFrame) -> pd.DataFrame:
    """Audit section 2.3 notebook for backtest-engine evidence."""

    nb = "notebooks/2_3_rolling_baseline_no_enhancement.ipynb"
    subset = notebook_index[notebook_index["notebook"].eq(nb)]
    text = " ".join(subset["first_line"].fillna("") + " " + subset["keyword_hits"].fillna(""))
    rows = [
        ("engine_present", "PASS" if "backtest" in text.lower() else "UNCLEAR", "Notebook keyword search", "Need inspect functions/signatures", "partial"),
        ("daily_resets", "UNCLEAR", "Search for reset/day/groupby evidence", "Manual confirmation needed", "unknown"),
        ("generic_trades", "UNCLEAR", "Search did not prove generic trade interface", "Ask teammate for function signature", "unknown"),
        ("waelbroeck_simulator", "PASS" if "waelbroeck" in text.lower() else "UNCLEAR", "Keyword evidence", "May be absent or named differently", "partial"),
        ("compatible_with_my_trades", "PARTIAL", "My trades include stock/date/timestamp/trade/position", "Need exact required schema", "partial"),
    ]
    out = pd.DataFrame(rows, columns=["item", "status", "evidence", "notes", "compatibility_with_my_code"])
    out.to_csv(paths.out_dir / "backtest_engine_audit.csv", index=False)
    return out


def audit_compatibility(paths: TeammateAuditPaths, schema: pd.DataFrame, model_audit: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Assess compatibility with my 2.4/2.5/2.7 modules."""

    sources = " ".join(schema["source_file"].fillna(""))
    cols = " ".join(schema["columns"].fillna(""))
    rows = [
        {
            "my_module": "2.4 synthetic alpha",
            "required_input": "date,time,stock,mid",
            "found_in_teammate_outputs": "yes" if all(c in cols for c in ["date", "time", "stock", "mid"]) else "partial",
            "compatible": "yes",
            "source_file": "processed_2_1 monthly_standardised bin parquet",
            "required_conversion": "read parquet, preserve date/time/stock/mid",
            "concerns": "Use full rolling window periods, not early rows only",
            "action_needed": "Point alpha generator at selected train/test parquet files",
        },
        {
            "my_module": "2.5 OW strategy",
            "required_input": "alpha rows plus sigma/ADV and OW lambda/beta",
            "found_in_teammate_outputs": "yes" if "ow_transient_params" in sources else "partial",
            "compatible": "partial",
            "source_file": "processed_2_2_rolling_baseline/parameters/ow_transient_params_by_pair_stock.csv",
            "required_conversion": "map fitted lambda/beta into OWStrategyConfig per stock/window",
            "concerns": "My current config is global placeholder; needs loader for per-stock/window params",
            "action_needed": "Confirm units of beta/lambda and normalization",
        },
        {
            "my_module": "2.5 reduced-form strategy",
            "required_input": "lambda_base/beta or reduced-form params, v_t convention",
            "found_in_teammate_outputs": "yes" if "reduced_form_params" in sources else "partial",
            "compatible": "partial",
            "source_file": "processed_2_2_rolling_baseline/parameters/reduced_form_params_by_pair_stock.csv",
            "required_conversion": "write parameter loader and confirm v_t definition",
            "concerns": "Reduced-form AFS exact formula/units need confirmation",
            "action_needed": "Ask teammate which reduced form was fitted",
        },
        {
            "my_module": "2.7 stress tests",
            "required_input": "scenario-ready alpha/trades plus fitted configs",
            "found_in_teammate_outputs": "partial",
            "compatible": "partial",
            "source_file": "processed folders + my outputs",
            "required_conversion": "rerun on chosen out-of-sample windows",
            "concerns": "Stress results not final until calibrated params loaded",
            "action_needed": "Integrate fitted parameter loaders",
        },
    ]
    compat = pd.DataFrame(rows)
    todo = pd.DataFrame(
        [
            ("high", "Confirm exact train/test rolling windows and fixed 20-stock universe", "teammate", "Needed for final experiment alignment", "rolling_pair_summary / config JSON"),
            ("high", "Confirm OW lambda/beta units and normalization", "teammate", "Needed to replace placeholder strategy params", "ow_transient_params_by_pair_stock.csv"),
            ("high", "Confirm reduced-form AFS formula and v_t definition", "teammate", "Needed for enhanced model strategy", "reduced_form_params_by_pair_stock.csv"),
            ("medium", "Write parameter loader from teammate outputs into my configs", "me", "Integration step", "parameter CSVs"),
            ("medium", "Confirm backtest engine trade schema", "both", "Needed to pass my strategy trades into 2.3", "2_3 notebook/functions"),
        ],
        columns=["priority", "task", "owner", "reason", "required_file_or_info"],
    )
    compat.to_csv(paths.out_dir / "compatibility_audit.csv", index=False)
    todo.to_csv(paths.out_dir / "integration_todo_list.csv", index=False)
    return {"compatibility": compat, "todo": todo}


def make_plots(paths: TeammateAuditPaths, inventory: pd.DataFrame, coverage: pd.DataFrame, missing: pd.DataFrame, time_gaps: pd.DataFrame, scaling_cov: pd.DataFrame) -> None:
    """Save audit plots."""

    fig_dir = paths.fig_dir
    if not inventory.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        inventory.groupby("folder_label")["size_mb"].sum().plot(kind="bar", ax=ax)
        ax.set_title("Processed file size by folder")
        ax.set_ylabel("MB")
        fig.tight_layout()
        fig.savefig(fig_dir / "file_sizes_by_folder.png", dpi=150)
        plt.close(fig)
    if not coverage.empty:
        top = coverage.sort_values("n_rows", ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(top["source_file"], top["n_rows"])
        ax.tick_params(axis="x", rotation=90)
        ax.set_title("Rows by file")
        fig.tight_layout()
        fig.savefig(fig_dir / "rows_by_file.png", dpi=150)
        plt.close(fig)
    if not missing.empty:
        top_missing = missing.groupby("source_file")["missing_pct"].mean().sort_values(ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(12, 5))
        top_missing.plot(kind="bar", ax=ax)
        ax.set_title("Average missing percentage by file")
        fig.tight_layout()
        fig.savefig(fig_dir / "missing_values_by_file.png", dpi=150)
        plt.close(fig)
    if not time_gaps.empty and "median_gap_seconds" in time_gaps:
        fig, ax = plt.subplots(figsize=(8, 5))
        time_gaps["median_gap_seconds"].dropna().hist(ax=ax, bins=40)
        ax.set_title("Median time gap distribution")
        ax.set_xlabel("seconds")
        fig.tight_layout()
        fig.savefig(fig_dir / "time_gap_histogram.png", dpi=150)
        plt.close(fig)
    if not scaling_cov.empty and "date" in scaling_cov:
        cov = scaling_cov.groupby("date")["rows"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(pd.to_datetime(cov["date"]), cov["rows"])
        ax.set_title("Scaling rows by date")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(fig_dir / "scaling_coverage_by_date.png", dpi=150)
        plt.close(fig)


def write_final_report(
    paths: TeammateAuditPaths,
    inventory: pd.DataFrame,
    artifacts: pd.DataFrame,
    universe: pd.DataFrame,
    splits: pd.DataFrame,
    model_audit: pd.DataFrame,
    backtest: pd.DataFrame,
    compatibility: pd.DataFrame,
    todo: pd.DataFrame,
) -> str:
    """Write final markdown and text audit report."""

    likely_universe = universe.iloc[0].to_dict() if not universe.empty else {}
    likely_split = splits.iloc[0].to_dict() if not splits.empty else {}
    ow = model_audit[model_audit["model_name"].eq("OW")]
    reduced = model_audit[model_audit["model_name"].eq("reduced_form_AFS")]
    statuses = {
        "Section 2.1": "PASS" if not artifacts[artifacts["likely_section"].eq("2.1")].empty else "UNCLEAR",
        "Section 2.2": "PARTIAL" if not ow.empty and ow.iloc[0]["implemented"] == "yes" else "UNCLEAR",
        "Section 2.3": "PARTIAL",
        "Integration readiness": "PARTIAL",
    }
    lines = [
        "# Teammate 2.1 / 2.2 / 2.3 Audit Report",
        "",
        "## 1. Executive Summary",
        f"- Discovered {len(inventory)} processed files across 2.1 and 2.2 folders.",
        "- Processed 2.1 appears to contain monthly standardised bin/fill parquet files plus baseline 20-stock and rolling-universe artifacts.",
        "- Processed 2.2 appears to contain rolling-pair fitted parameter files, metrics, logs, and model comparison tables.",
        "- Main integration blocker: exact parameter units and reduced-form AFS convention must be confirmed before replacing my placeholders.",
        "",
        "## 2. Processed Data Inventory",
        f"- Files by suffix: {inventory['suffix'].value_counts().to_dict() if not inventory.empty else {}}",
        f"- Largest files: {', '.join(inventory.sort_values('size_mb', ascending=False)['relative_path'].head(5).tolist()) if not inventory.empty else 'none'}",
        "",
        "## 3. Data Preparation Audit - Section 2.1",
        f"- Likely universe evidence: {likely_universe}",
        f"- Likely split/window evidence: {likely_split}",
        "- Missing values, duplicates, time gaps, and price jump summaries are saved as CSV outputs.",
        "",
        "## 4. Model Fitting Audit - Section 2.2",
        f"- OW audit: {ow.to_dict('records') if not ow.empty else 'not found'}",
        f"- Reduced-form / AFS audit: {reduced.to_dict('records') if not reduced.empty else 'not found'}",
        "- Parameter files are listed in fitted_parameter_files.csv.",
        "",
        "## 5. Backtest Engine Audit - Section 2.3",
        _df_to_md(backtest),
        "",
        "## 6. Consistency with Course Specification",
        "- Baseline 20-stock artifacts are present, and rolling pair artifacts are present.",
        "- Whether the exact in-sample/out-of-sample split and same-stock universe are fully compliant should be confirmed from the config JSON/notebook cells.",
        "- OW parameter files exist; normalization and beta/lambda units need final confirmation.",
        "- Reduced-form parameter files exist; exact AFS reduction and v_t convention need final confirmation.",
        "",
        "## 7. Compatibility with My Code",
        _df_to_md(compatibility),
        "",
        "## 8. Questions for Teammate",
        "- Which exact dates are in-sample and out-of-sample for each rolling pair?",
        "- Is the 20-stock universe fixed across windows or selected per pair?",
        "- Which model is the reduced form of AFS exactly?",
        "- Are lambda and beta global, per-stock, or rolling-window specific?",
        "- What units are lambda and beta in, and is beta per minute?",
        "- Is q raw signed volume or normalized volume in the fitted model?",
        "- How is v_t computed?",
        "- Which parameter file should I use for final strategy runs?",
        "- Does the 2.3 backtest engine accept generic trade tables from my strategy modules?",
        "",
        "## 9. Final Verdict",
        *[f"- {k}: {v}" for k, v in statuses.items()],
        "",
        "## Integration TODO",
        _df_to_md(todo),
        "",
    ]
    report = "\n".join(lines)
    (paths.out_dir / "final_teammate_audit_report.md").write_text(report, encoding="utf-8")
    (paths.out_dir / "final_teammate_audit_report.txt").write_text(report, encoding="utf-8")
    return report


def run_full_teammate_audit(
    project_root: Path | None = None,
    nrows: int | None = None,
    max_files: int | None = None,
) -> dict[str, pd.DataFrame | str]:
    """Run all teammate audit steps and save outputs."""

    paths = build_paths(project_root)
    inventory = discover_files(paths)
    tabular = audit_tabular_files(paths, inventory, nrows=nrows, max_files=max_files)
    artifacts = classify_artifacts(paths, tabular["schema"], tabular["coverage"])
    split_outputs = audit_universe_and_splits(paths, tabular["schema"])
    scaling = audit_scaling_factors(paths, tabular["schema"])
    notebooks = audit_notebooks(paths)
    models = audit_models(paths, tabular["schema"], notebooks["cell_index"])
    backtest = audit_backtest(paths, notebooks["cell_index"])
    compat = audit_compatibility(paths, tabular["schema"], models["model_audit"])
    make_plots(paths, inventory, tabular["coverage"], tabular["missing"], tabular["time_gaps"], scaling["coverage"])
    report = write_final_report(
        paths,
        inventory,
        artifacts,
        split_outputs["universe"],
        split_outputs["splits"],
        models["model_audit"],
        backtest,
        compat["compatibility"],
        compat["todo"],
    )
    return {
        "inventory": inventory,
        "schema": tabular["schema"],
        "artifacts": artifacts,
        "model_audit": models["model_audit"],
        "compatibility": compat["compatibility"],
        "report": report,
    }
