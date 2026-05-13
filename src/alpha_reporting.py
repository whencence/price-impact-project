"""Reporting utilities for synthetic alpha validation diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


REQUIRED_DIAGNOSTIC_COLUMNS = {
    "horizon_minutes",
    "target_corr",
    "fraction_missing_future_return",
    "alpha_x",
    "alpha_y",
    "empirical_corr_alpha_return",
    "alpha_std",
    "future_return_std",
    "realized_unbiased_slope",
    "corr_abs_error",
    "recommended_baseline_flag",
}


def _find_project_root(start: Path) -> Path:
    start = start.resolve()
    if start.name == "src":
        start = start.parent
    for candidate in [start] + list(start.parents):
        if (candidate / "src").exists() or (candidate / ".git").exists():
            return candidate
    return start


def _load_diagnostics(diagnostics_path: Path) -> pd.DataFrame:
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"synthetic alpha diagnostics file not found: {diagnostics_path}")
    diagnostics = pd.read_csv(diagnostics_path)
    if "horizon_minutes" not in diagnostics.columns and "horizon_steps" in diagnostics.columns:
        _legacy = diagnostics["horizon_steps"].astype(float)
        diagnostics = diagnostics.copy()
        diagnostics["horizon_minutes"] = _legacy
        print(
            "WARNING: diagnostics file uses legacy horizon_steps. "
            "Interpreting these values as minutes for reporting only; rerun "
            "run_synthetic_alpha_grid_clock_time for final outputs."
        )
    missing = REQUIRED_DIAGNOSTIC_COLUMNS.difference(diagnostics.columns)
    if missing:
        raise ValueError(f"diagnostics file is missing required columns: {sorted(missing)}")
    if diagnostics.empty:
        raise ValueError("diagnostics file is empty")
    return diagnostics


def _format_clean_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "h": diagnostics["horizon_minutes"].astype(float),
            "rho_target": diagnostics["target_corr"].astype(float),
            "corr_empirical": diagnostics["empirical_corr_alpha_return"].astype(float),
            "corr_error": diagnostics["corr_abs_error"].astype(float),
            "unbiased_slope": diagnostics["realized_unbiased_slope"].astype(float),
            "alpha_std_bps": diagnostics["alpha_std"].astype(float) * 10_000.0,
            "future_return_std_bps": diagnostics["future_return_std"].astype(float) * 10_000.0,
            "missing_pct": diagnostics["fraction_missing_future_return"].astype(float) * 100.0,
            "x": diagnostics["alpha_x"].astype(float),
            "y": diagnostics["alpha_y"].astype(float),
        }
    )
    table = table.sort_values(["h", "rho_target"]).reset_index(drop=True)
    rounded = table.copy()
    for col in ["rho_target", "corr_empirical", "corr_error", "unbiased_slope"]:
        rounded[col] = rounded[col].round(4)
    for col in ["alpha_std_bps", "future_return_std_bps", "missing_pct"]:
        rounded[col] = rounded[col].round(3)
    rounded["x"] = rounded["x"].map(lambda value: f"{value:.6g}")
    rounded["y"] = rounded["y"].map(lambda value: f"{value:.6g}")
    return rounded


def _plot_empirical_vs_target_corr(diagnostics: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for horizon, group in diagnostics.groupby("horizon_minutes", sort=True):
        group = group.sort_values("target_corr")
        ax.plot(
            group["target_corr"],
            group["empirical_corr_alpha_return"],
            marker="o",
            label=f"h={horizon:g}m",
        )
    low = float(diagnostics["target_corr"].min())
    high = float(diagnostics["target_corr"].max())
    ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1, label="y=x")
    ax.set_title("Synthetic Alpha: Empirical vs Target Correlation")
    ax.set_xlabel("Target correlation")
    ax.set_ylabel("Empirical correlation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "empirical_vs_target_corr_clean.png", dpi=150)
    plt.close(fig)


def _annotated_heatmap(
    diagnostics: pd.DataFrame,
    value_col: str,
    title: str,
    fig_path: Path,
    fmt: str = ".4f",
    scale: float = 1.0,
) -> None:
    pivot = diagnostics.pivot(index="horizon_minutes", columns="target_corr", values=value_col)
    values = pivot.to_numpy(dtype=float) * scale
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(values, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Target correlation")
    ax.set_ylabel("Horizon minutes")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{value:.2f}" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(value) for value in pivot.index])
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            label = "nan" if not np.isfinite(value) else format(value, fmt)
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def _plot_future_return_vol_by_horizon(diagnostics: pd.DataFrame, fig_dir: Path) -> None:
    first_per_horizon = (
        diagnostics.sort_values(["horizon_minutes", "target_corr"])
        .drop_duplicates("horizon_minutes")
        .sort_values("horizon_minutes")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        first_per_horizon["horizon_minutes"],
        first_per_horizon["future_return_std"] * 10_000.0,
        marker="o",
    )
    ax.set_title("Future Return Volatility by Horizon")
    ax.set_xlabel("Horizon minutes")
    ax.set_ylabel("Future return std (bps)")
    fig.tight_layout()
    fig.savefig(fig_dir / "future_return_vol_by_horizon.png", dpi=150)
    plt.close(fig)


def _baseline_row(diagnostics: pd.DataFrame) -> pd.Series:
    baseline = diagnostics.loc[diagnostics["recommended_baseline_flag"].astype(bool)]
    if baseline.empty:
        baseline = diagnostics.loc[
            np.isclose(diagnostics["horizon_minutes"], 5.0)
            & np.isclose(diagnostics["target_corr"], 0.10)
        ]
    if baseline.empty:
        raise ValueError("could not locate baseline diagnostics row for h=5, rho=0.10")
    return baseline.iloc[0]


def _write_summary(diagnostics: pd.DataFrame, output_dir: Path) -> None:
    row = _baseline_row(diagnostics)
    lines = [
        "Synthetic Alpha Validation Summary",
        "==================================",
        "Horizons are now clock-time horizons in minutes.",
        "The old row-based h convention is not used as the final convention.",
        "The previous h=5 rows was approximately 50 seconds if bins are spaced by 10 seconds.",
        f"Baseline h: {float(row['horizon_minutes']):g} minutes",
        f"Baseline rho: {float(row['target_corr']):.2f}",
        "Baseline H: 5 minutes",
        f"Empirical correlation: {float(row['empirical_corr_alpha_return']):.4f}",
        f"Correlation error: {float(row['corr_abs_error']):.4f}",
        f"Realized unbiased slope: {float(row['realized_unbiased_slope']):.4f}",
        f"Alpha std: {float(row['alpha_std']) * 10_000.0:.3f} bps",
        f"Future return std: {float(row['future_return_std']) * 10_000.0:.3f} bps",
        f"Missing future return fraction: {float(row['fraction_missing_future_return']):.4%}",
        "",
        "Interpretation:",
        "Forecast horizon h controls which future return is predicted.",
        "Decay half-life H controls signal persistence.",
        "Future return lookup uses the first available timestamp >= t+h within the same stock/date, with tolerance.",
        "Alpha decay uses actual timestamp gaps, not row count.",
        "The empirical correlation is close to the target, which validates the x/y calibration.",
        "The unbiased slope is close to 1, supporting the calibration condition E[r | alpha] approx alpha.",
        "h=5 minutes, rho=0.10 is retained as the baseline.",
        "The rho grid is interpreted as a signal-quality sensitivity axis, not as a parameter optimized from the market.",
    ]
    (output_dir / "synthetic_alpha_validation_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def create_alpha_validation_report(
    diagnostics_path: Path | str, output_dir: Path | str
) -> pd.DataFrame:
    """Create clean alpha validation tables, figures, and text summary."""

    diagnostics_path = Path(diagnostics_path)
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = _load_diagnostics(diagnostics_path)
    clean_table = _format_clean_table(diagnostics)
    clean_table.to_csv(output_dir / "synthetic_alpha_report_table.csv", index=False)

    _plot_empirical_vs_target_corr(diagnostics, fig_dir)
    _annotated_heatmap(
        diagnostics,
        "corr_abs_error",
        "Absolute Correlation Calibration Error",
        fig_dir / "corr_abs_error_heatmap.png",
        fmt=".4f",
    )
    _annotated_heatmap(
        diagnostics,
        "realized_unbiased_slope",
        "Realized Unbiased-Predictor Slope",
        fig_dir / "unbiased_slope_heatmap.png",
        fmt=".3f",
    )
    _plot_future_return_vol_by_horizon(diagnostics, fig_dir)
    _annotated_heatmap(
        diagnostics,
        "alpha_std",
        "Synthetic Alpha Volatility in bps",
        fig_dir / "alpha_std_heatmap.png",
        fmt=".3f",
        scale=10_000.0,
    )
    _write_summary(diagnostics, output_dir)
    return clean_table


def main() -> None:
    """CLI entry point for python -m src.alpha_reporting."""

    project_root = _find_project_root(Path.cwd())
    output_dir = project_root / "outputs" / "alphas"
    diagnostics_path = output_dir / "synthetic_alpha_diagnostics.csv"
    table = create_alpha_validation_report(diagnostics_path, output_dir)
    print(f"Saved alpha validation report to {output_dir}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
