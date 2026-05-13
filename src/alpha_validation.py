"""Final validation checks for section 2.4 synthetic alphas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


EXPECTED_FILES = {
    "diagnostics": "synthetic_alpha_diagnostics.csv",
    "baseline": "synthetic_alpha_baseline_h5m_rho010.csv",
    "decay_diagnostics": "synthetic_alpha_decay_diagnostics.csv",
    "decay_metadata": "synthetic_alpha_decay_metadata.csv",
}

BASELINE_REQUIRED_COLUMNS = {
    "date",
    "time",
    "stock",
    "timestamp",
    "mid",
    "future_timestamp_h",
    "target_timestamp_h",
    "future_mid_h",
    "future_return_h",
    "future_time_gap_seconds",
    "valid_future_return",
    "alpha_synthetic",
    "alpha_x",
    "alpha_y",
    "target_corr",
    "horizon_minutes",
    "delta_w",
    "alpha_state_H1m",
    "alpha_state_H5m",
    "alpha_state_H30m",
    "alpha_state_H60m",
    "baseline_alpha_for_strategy",
}

DIAGNOSTIC_REQUIRED_COLUMNS = {
    "horizon_minutes",
    "target_corr",
    "n_rows_total",
    "n_valid_future_return",
    "fraction_missing_future_return",
    "var_future_return",
    "mean_inv_price_sq",
    "alpha_x",
    "alpha_y",
    "empirical_corr_alpha_return",
    "alpha_mean",
    "alpha_std",
    "future_return_mean",
    "future_return_std",
    "realized_unbiased_slope",
    "corr_abs_error",
    "recommended_baseline_flag",
}

STATE_COLUMNS = ["alpha_state_H1m", "alpha_state_H5m", "alpha_state_H30m", "alpha_state_H60m"]


@dataclass
class CheckResult:
    """Result for one validation check group."""

    name: str
    passed: bool
    details: str


def find_project_root(start: Path) -> Path:
    """Resolve project root from repo root or notebooks directory."""

    start = start.resolve()
    if start.name == "notebooks":
        start = start.parent
    for candidate in [start] + list(start.parents):
        if (candidate / "src").exists() and ((candidate / "outputs").exists() or (candidate / ".git").exists()):
            return candidate
    raise RuntimeError("Could not find project root containing src/ and outputs/ or .git/")


def expected_alpha_paths(output_dir: Path) -> dict[str, Path]:
    """Return expected alpha output paths."""

    return {name: output_dir / filename for name, filename in EXPECTED_FILES.items()}


def load_alpha_validation_inputs(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Load diagnostics, baseline alpha, decay diagnostics, and decay metadata."""

    paths = expected_alpha_paths(output_dir)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required alpha validation files:\n" + "\n".join(missing))
    return {
        "diagnostics": pd.read_csv(paths["diagnostics"]),
        "baseline": pd.read_csv(paths["baseline"], parse_dates=["timestamp", "target_timestamp_h", "future_timestamp_h"]),
        "decay_diagnostics": pd.read_csv(paths["decay_diagnostics"]),
        "decay_metadata": pd.read_csv(paths["decay_metadata"]),
    }


def make_clean_alpha_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Create a compact report table for alpha diagnostics."""

    return pd.DataFrame(
        {
            "h_minutes": diagnostics["horizon_minutes"].astype(float),
            "rho_target": diagnostics["target_corr"].astype(float),
            "corr_empirical": diagnostics["empirical_corr_alpha_return"].astype(float).round(4),
            "corr_error": diagnostics["corr_abs_error"].astype(float).round(4),
            "unbiased_slope": diagnostics["realized_unbiased_slope"].astype(float).round(4),
            "alpha_std_bps": (diagnostics["alpha_std"].astype(float) * 10_000.0).round(3),
            "future_return_std_bps": (diagnostics["future_return_std"].astype(float) * 10_000.0).round(3),
            "missing_pct": (diagnostics["fraction_missing_future_return"].astype(float) * 100.0).round(3),
        }
    ).sort_values(["h_minutes", "rho_target"]).reset_index(drop=True)


def make_clean_decay_table(decay_diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Create a compact report table for alpha decay diagnostics."""

    return pd.DataFrame(
        {
            "H_minutes": decay_diagnostics["half_life_minutes"].astype(float),
            "output_col": decay_diagnostics["output_col"],
            "corr_raw_alpha_future_return": decay_diagnostics["corr_raw_alpha_future_return"].astype(float).round(4),
            "corr_alpha_state_future_return": decay_diagnostics["corr_alpha_state_future_return"].astype(float).round(4),
            "raw_alpha_std_bps": decay_diagnostics["raw_alpha_std_bps"].astype(float).round(3),
            "alpha_state_std_bps": decay_diagnostics["alpha_state_std_bps"].astype(float).round(3),
            "state_to_raw_std_ratio": decay_diagnostics["state_to_raw_std_ratio"].astype(float).round(4),
        }
    ).sort_values("H_minutes").reset_index(drop=True)


def validate_file_and_schema(
    diagnostics: pd.DataFrame,
    baseline: pd.DataFrame,
    decay_diagnostics: pd.DataFrame,
    decay_metadata: pd.DataFrame,
) -> CheckResult:
    """Validate files, required columns, scenario counts, and baseline rows."""

    errors = []
    missing_baseline = BASELINE_REQUIRED_COLUMNS.difference(baseline.columns)
    missing_diag = DIAGNOSTIC_REQUIRED_COLUMNS.difference(diagnostics.columns)
    if missing_baseline:
        errors.append(f"baseline missing columns: {sorted(missing_baseline)}")
    if missing_diag:
        errors.append(f"diagnostics missing columns: {sorted(missing_diag)}")
    expected_pairs = {(h, r) for h in [1.0, 5.0, 10.0] for r in [0.05, 0.10, 0.20, 0.30, 0.50]}
    observed_pairs = set(zip(diagnostics.get("horizon_minutes", []), diagnostics.get("target_corr", [])))
    if observed_pairs != expected_pairs:
        errors.append(f"expected 15 h/rho pairs, got {len(observed_pairs)}")
    baseline_rows = diagnostics.loc[
        np.isclose(diagnostics.get("horizon_minutes", np.nan), 5.0)
        & np.isclose(diagnostics.get("target_corr", np.nan), 0.10)
        & diagnostics.get("recommended_baseline_flag", False).astype(bool)
    ]
    if baseline_rows.empty:
        errors.append("baseline diagnostics row h=5.0, rho=0.10, recommended=True not found")
    expected_h = {1.0, 5.0, 30.0, 60.0}
    observed_h = set(decay_diagnostics.get("half_life_minutes", []))
    if observed_h != expected_h:
        errors.append(f"decay diagnostics half-lives mismatch: {sorted(observed_h)}")
    if not {"half_life_minutes", "output_col"}.issubset(decay_metadata.columns):
        errors.append("decay metadata missing half_life_minutes/output_col")
    return CheckResult("file/schema", not errors, "; ".join(errors) if errors else "All required files, columns, scenarios, and metadata found.")


def validate_clock_time_horizons(
    baseline: pd.DataFrame, tolerance_seconds: float = 30.0
) -> tuple[CheckResult, pd.DataFrame]:
    """Validate target/future timestamp construction for baseline alpha."""

    errors = []
    work = baseline.copy()
    valid = work.loc[work["valid_future_return"].astype(bool)].copy()
    if not np.allclose(work["horizon_minutes"].dropna().astype(float), 5.0):
        errors.append("horizon_minutes is not 5.0 for all baseline rows")
    expected_target = work["timestamp"] + pd.to_timedelta(5.0, unit="m")
    target_error = (work["target_timestamp_h"] - expected_target).dt.total_seconds().abs().fillna(0)
    if target_error.max() > 1e-6:
        errors.append(f"target_timestamp_h mismatch max seconds={target_error.max()}")
    if not valid.empty:
        if (valid["future_timestamp_h"] < valid["target_timestamp_h"]).any():
            errors.append("future_timestamp_h before target_timestamp_h on valid rows")
        if (valid["future_time_gap_seconds"] < -1e-9).any():
            errors.append("negative future_time_gap_seconds on valid rows")
        if (valid["future_time_gap_seconds"] > tolerance_seconds + 1e-9).any():
            errors.append("future_time_gap_seconds exceeds tolerance on valid rows")
        if (valid["future_timestamp_h"].dt.date.astype(str) != valid["date"].astype(str)).any():
            errors.append("future timestamp crosses date on valid rows")
    missing_fraction = 1.0 - len(valid) / len(work) if len(work) else 1.0
    if missing_fraction >= 0.05:
        errors.append(f"fraction missing future return is {missing_fraction:.2%}, expected < 5%")
    gap_summary = valid["future_time_gap_seconds"].describe(percentiles=[0.5, 0.95, 0.99]).to_frame("future_time_gap_seconds")
    return CheckResult("clock-time horizons", not errors, "; ".join(errors) if errors else "Clock-time horizon construction passed."), gap_summary


def validate_raw_alpha_calibration(diagnostics: pd.DataFrame) -> tuple[CheckResult, pd.DataFrame]:
    """Validate empirical correlation, unbiased slope, and missingness thresholds."""

    failures = diagnostics.loc[
        (diagnostics["corr_abs_error"] >= 0.01)
        | ((diagnostics["realized_unbiased_slope"] - 1.0).abs() >= 0.05)
        | (diagnostics["fraction_missing_future_return"] >= 0.10)
    ].copy()
    passed = failures.empty
    details = "All h/rho calibration checks passed." if passed else f"{len(failures)} h/rho rows failed thresholds."
    return CheckResult("raw alpha calibration", passed, details), failures


def validate_alpha_decay(
    baseline: pd.DataFrame, decay_diagnostics: pd.DataFrame
) -> CheckResult:
    """Validate state columns, reset convention, and baseline strategy alpha."""

    errors = []
    missing = [col for col in STATE_COLUMNS + ["baseline_alpha_for_strategy"] if col not in baseline.columns]
    if missing:
        errors.append(f"missing state columns: {missing}")
        return CheckResult("alpha decay", False, "; ".join(errors))
    if not np.allclose(baseline["baseline_alpha_for_strategy"], baseline["alpha_state_H5m"], equal_nan=False):
        errors.append("baseline_alpha_for_strategy differs from alpha_state_H5m")
    for col in STATE_COLUMNS:
        if baseline[col].isna().any():
            errors.append(f"{col} contains NaNs")
        if np.allclose(baseline[col].to_numpy(), 0.0):
            errors.append(f"{col} is all zero")
    first_rows = baseline.sort_values(["stock", "date", "timestamp"]).groupby(["stock", "date"], sort=False).head(1)
    for col in STATE_COLUMNS:
        if not np.allclose(first_rows[col], first_rows["alpha_synthetic"], atol=1e-14):
            errors.append(f"{col} does not reset to alpha_synthetic at group starts")
    std_h1 = float(baseline["alpha_state_H1m"].std(ddof=1))
    std_h60 = float(baseline["alpha_state_H60m"].std(ddof=1))
    if std_h60 > std_h1:
        errors.append("warning: alpha_state_H60m std exceeds alpha_state_H1m std")
    return CheckResult("alpha decay", not [e for e in errors if not e.startswith("warning")], "; ".join(errors) if errors else "Alpha decay checks passed.")


def validate_brownian_noise(baseline: pd.DataFrame) -> tuple[CheckResult, pd.DataFrame]:
    """Validate DeltaW mean and variance against horizon_minutes."""

    delta_w = baseline["delta_w"].dropna().astype(float)
    horizon = float(baseline["horizon_minutes"].dropna().iloc[0])
    mean = float(delta_w.mean())
    var = float(delta_w.var(ddof=1))
    summary = pd.DataFrame(
        [{"delta_w_mean": mean, "delta_w_var": var, "expected_delta_w_var": horizon}]
    )
    passed = abs(mean) < 0.05 * np.sqrt(horizon) and abs(var - horizon) / horizon < 0.10
    details = "Brownian noise checks passed." if passed else "DeltaW mean/variance outside tolerance."
    return CheckResult("Brownian noise", passed, details), summary


def validate_formula_consistency(baseline: pd.DataFrame) -> tuple[CheckResult, pd.DataFrame]:
    """Reconstruct alpha formula and compare against stored alpha_synthetic."""

    valid = baseline.loc[baseline["valid_future_return"].astype(bool)].copy()
    reconstructed = (
        valid["alpha_x"] * valid["future_return_h"] + valid["alpha_y"] * valid["delta_w"] / valid["mid"]
    )
    error = (reconstructed - valid["alpha_synthetic"]).abs()
    summary = pd.DataFrame(
        [{"max_abs_error": float(error.max()), "mean_abs_error": float(error.mean())}]
    )
    passed = float(error.max()) < 1e-12
    details = "Formula reconstruction matches alpha_synthetic." if passed else "Formula reconstruction error above tolerance."
    return CheckResult("formula consistency", passed, details), summary


def create_strategy_alpha_input(baseline: pd.DataFrame, output_dir: Path) -> tuple[CheckResult, pd.DataFrame]:
    """Create cleaned strategy input CSV for section 2.5."""

    required = [
        "date",
        "time",
        "timestamp",
        "stock",
        "mid",
        "alpha_synthetic",
        "baseline_alpha_for_strategy",
        "future_return_h",
        "valid_future_return",
    ]
    missing = [col for col in required if col not in baseline.columns]
    if missing:
        return CheckResult("strategy input readiness", False, f"missing columns: {missing}"), pd.DataFrame()
    cols = required + [col for col in ["spread", "depth", "source_file"] if col in baseline.columns]
    strategy = baseline.loc[:, cols].rename(
        columns={
            "alpha_synthetic": "alpha_raw",
            "baseline_alpha_for_strategy": "alpha_for_strategy",
        }
    )
    path = output_dir / "strategy_alpha_input_h5m_rho010_H5m.csv"
    strategy.to_csv(path, index=False)
    return CheckResult("strategy input readiness", True, f"Saved {path}"), strategy


def plot_validation_figures(
    diagnostics: pd.DataFrame,
    baseline: pd.DataFrame,
    decay_diagnostics: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Save validation plots under figures/validation."""

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for h, group in diagnostics.groupby("horizon_minutes", sort=True):
        group = group.sort_values("target_corr")
        ax.plot(group["target_corr"], group["empirical_corr_alpha_return"], marker="o", label=f"h={h:g}m")
    low, high = diagnostics["target_corr"].min(), diagnostics["target_corr"].max()
    ax.plot([low, high], [low, high], linestyle="--", color="black")
    ax.set_title("Empirical vs target correlation by horizon")
    ax.set_xlabel("Target rho")
    ax.set_ylabel("Empirical correlation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "validation_empirical_vs_target_corr.png", dpi=150)
    plt.close(fig)

    for value_col, filename, title in [
        ("corr_abs_error", "validation_corr_abs_error_heatmap.png", "Correlation absolute error"),
        ("realized_unbiased_slope", "validation_unbiased_slope_heatmap.png", "Unbiased slope"),
    ]:
        pivot = diagnostics.pivot(index="horizon_minutes", columns="target_corr", values=value_col)
        values = pivot.to_numpy()
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(values, aspect="auto", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("rho")
        ax.set_ylabel("h minutes")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{x:g}" for x in pivot.index])
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=150)
        plt.close(fig)

    one = diagnostics.sort_values(["horizon_minutes", "target_corr"]).drop_duplicates("horizon_minutes")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(one["horizon_minutes"], one["future_return_std"] * 10_000.0, marker="o")
    ax.set_title("Future return std by horizon")
    ax.set_xlabel("h minutes")
    ax.set_ylabel("std bps")
    fig.tight_layout()
    fig.savefig(fig_dir / "validation_future_return_std_bps.png", dpi=150)
    plt.close(fig)

    sample_key = baseline.groupby(["stock", "date"], sort=False).size().idxmax()
    sample = baseline.loc[(baseline["stock"] == sample_key[0]) & (baseline["date"] == sample_key[1])].head(500)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(sample["timestamp"], sample["alpha_synthetic"], label="alpha_synthetic", linewidth=1)
    for col in STATE_COLUMNS:
        ax.plot(sample["timestamp"], sample[col], label=col, linewidth=1)
    ax.set_title("Alpha raw/state sample path")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("alpha")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "validation_alpha_state_sample_path.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(decay_diagnostics["half_life_minutes"], decay_diagnostics["corr_alpha_state_future_return"], marker="o")
    ax.set_xscale("log")
    ax.set_title("Decay correlation by half-life")
    ax.set_xlabel("H minutes")
    ax.set_ylabel("corr(state, future return)")
    fig.tight_layout()
    fig.savefig(fig_dir / "validation_decay_corr_by_half_life.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(decay_diagnostics["half_life_minutes"], decay_diagnostics["alpha_state_std_bps"], marker="o")
    ax.set_xscale("log")
    ax.set_title("Alpha state std by half-life")
    ax.set_xlabel("H minutes")
    ax.set_ylabel("state std bps")
    fig.tight_layout()
    fig.savefig(fig_dir / "validation_alpha_state_std_by_half_life.png", dpi=150)
    plt.close(fig)


def write_final_validation_summary(
    output_dir: Path,
    results: list[CheckResult],
    diagnostics: pd.DataFrame,
    decay_diagnostics: pd.DataFrame,
) -> None:
    """Write final section 2.4 pass/fail summary."""

    baseline = diagnostics.loc[diagnostics["recommended_baseline_flag"].astype(bool)].iloc[0]
    h5 = decay_diagnostics.loc[np.isclose(decay_diagnostics["half_life_minutes"], 5.0)].iloc[0]
    passed_all = all(result.passed for result in results)
    lines = [
        "Final Section 2.4 Validation Summary",
        "====================================",
        "",
        "Check status:",
    ]
    lines.extend([f"- {r.name}: {'PASS' if r.passed else 'FAIL'} - {r.details}" for r in results])
    lines.extend(
        [
            "",
            "Baseline values:",
            "h = 5 minutes",
            "rho = 0.10",
            "H = 5 minutes",
            f"empirical corr = {baseline['empirical_corr_alpha_return']:.6f}",
            f"corr error = {baseline['corr_abs_error']:.6f}",
            f"unbiased slope = {baseline['realized_unbiased_slope']:.6f}",
            f"future return std bps = {baseline['future_return_std'] * 10_000.0:.3f}",
            f"alpha std bps = {baseline['alpha_std'] * 10_000.0:.3f}",
            f"fraction missing future return = {baseline['fraction_missing_future_return']:.6f}",
            f"corr alpha_state_H5m vs future return = {h5['corr_alpha_state_future_return']:.6f}",
            f"alpha_state_H5m std bps = {h5['alpha_state_std_bps']:.3f}",
            "",
            "Conclusion:",
            "Section 2.4 is ready for section 2.5." if passed_all else "Section 2.4 has validation failures to review before section 2.5.",
        ]
    )
    (output_dir / "final_section_2_4_validation_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_final_alpha_validation(output_dir: Path) -> dict[str, Any]:
    """Run all final section 2.4 validation checks and save outputs."""

    output_dir = Path(output_dir)
    data = load_alpha_validation_inputs(output_dir)
    diagnostics = data["diagnostics"]
    baseline = data["baseline"]
    decay_diagnostics = data["decay_diagnostics"]
    decay_metadata = data["decay_metadata"]

    clean_alpha = make_clean_alpha_table(diagnostics)
    clean_decay = make_clean_decay_table(decay_diagnostics)
    clean_alpha.to_csv(output_dir / "final_alpha_validation_table.csv", index=False)
    clean_decay.to_csv(output_dir / "final_alpha_decay_validation_table.csv", index=False)

    results: list[CheckResult] = []
    results.append(validate_file_and_schema(diagnostics, baseline, decay_diagnostics, decay_metadata))
    clock_result, gap_summary = validate_clock_time_horizons(baseline)
    results.append(clock_result)
    calibration_result, calibration_failures = validate_raw_alpha_calibration(diagnostics)
    results.append(calibration_result)
    results.append(validate_alpha_decay(baseline, decay_diagnostics))
    brownian_result, brownian_summary = validate_brownian_noise(baseline)
    results.append(brownian_result)
    formula_result, formula_summary = validate_formula_consistency(baseline)
    results.append(formula_result)
    strategy_result, strategy_input = create_strategy_alpha_input(baseline, output_dir)
    results.append(strategy_result)

    fig_dir = output_dir / "figures" / "validation"
    plot_validation_figures(diagnostics, baseline, decay_diagnostics, fig_dir)
    write_final_validation_summary(output_dir, results, diagnostics, decay_diagnostics)

    return {
        "results": pd.DataFrame([result.__dict__ for result in results]),
        "clean_alpha_table": clean_alpha,
        "clean_decay_table": clean_decay,
        "gap_summary": gap_summary,
        "calibration_failures": calibration_failures,
        "brownian_summary": brownian_summary,
        "formula_summary": formula_summary,
        "strategy_input_head": strategy_input.head(),
    }

