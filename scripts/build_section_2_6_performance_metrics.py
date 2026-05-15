from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/full_runs/full_allpairs_baseline_proxy"
OUT = ROOT / "outputs/report_tables"
STRATEGY = "OW_transient_proxy"


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"missing required file: {path}")
    return pd.read_csv(path)


def parse_alpha_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    match = re.search(r"^##+\s*[^\n]*Alpha Diagnostics[^\n]*\n", text, re.I | re.M)
    if not match:
        return {}
    next_section = re.search(r"^##\s+", text[match.end() :], re.M)
    end = match.end() + next_section.start() if next_section else len(text)
    section = text[match.start() : end]

    def metric(label: str) -> float:
        found = re.search(
            rf"{re.escape(label)}\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            section,
            re.I,
        )
        return float(found.group(1)) if found else np.nan

    return {
        "corr_alpha_future_return": metric("corr(alpha, future_return_h)"),
        "corr_trade_alpha": metric("corr(trade, alpha)"),
        "share_sign_trade_matches_alpha": metric("share_sign_trade_matches_alpha"),
        "gross_alpha_capture": metric("gross alpha capture"),
    }


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def fmt_m(value: float) -> str:
    return f"{value / 1_000_000:.2f}"


def fmt_bn(value: float) -> str:
    return f"{value / 1_000_000_000:.2f}"


def fmt_signed_m(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}\\${abs(value) / 1_000_000:.2f}m"


def fmt_part(value: float) -> str:
    pct = 100 * value
    return f"{pct:.4f}\\%" if abs(pct) < 0.1 else f"{pct:.3f}\\%"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    strategy = read_csv_required(BASE / "all_pairs_strategy_summary.csv")
    fitted = read_csv_required(BASE / "all_pairs_fitted_proxy_summary.csv")

    if "strategy_model" not in strategy.columns or "strategy_model" not in fitted.columns:
        fail("strategy_model column missing from baseline summaries")
    if not strategy["strategy_model"].eq(STRATEGY).all():
        fail("all_pairs_strategy_summary.csv contains non-OW_transient_proxy rows")
    if not fitted["strategy_model"].eq(STRATEGY).all():
        fail("all_pairs_fitted_proxy_summary.csv contains non-OW_transient_proxy rows")

    required_strategy_cols = [
        "pair_id",
        "number_of_stock_days",
        "mean_daily_net_pnl",
        "annualized_sharpe",
        "total_notional_turnover",
        "max_drawdown",
        "max_participation_rate",
        "mean_participation_rate",
        "max_abs_impact",
    ]
    required_fitted_cols = [
        "pair_id",
        "total_net_pnl_fitted_proxy",
        "total_fitted_proxy_cost",
    ]
    missing = [
        c
        for c in required_strategy_cols
        if c not in strategy.columns
    ] + [c for c in required_fitted_cols if c not in fitted.columns]
    if missing:
        fail(f"missing required columns: {missing}")

    strategy_pairs = set(strategy["pair_id"].astype(int))
    fitted_pairs = set(fitted["pair_id"].astype(int))
    if strategy_pairs != fitted_pairs:
        fail(f"pair mismatch between strategy and fitted summaries: {strategy_pairs} vs {fitted_pairs}")
    if strategy["pair_id"].duplicated().any() or fitted["pair_id"].duplicated().any():
        fail("duplicate pair_id values found")

    finite_cols = ["mean_daily_net_pnl", "annualized_sharpe"]
    if not np.isfinite(strategy[finite_cols].to_numpy(dtype=float)).all():
        fail("non-finite net PnL or Sharpe values found")
    if not np.isfinite(fitted[["total_net_pnl_fitted_proxy"]].to_numpy(dtype=float)).all():
        fail("non-finite fitted proxy net PnL values found")

    alpha = parse_alpha_metrics(BASE / "reports/integrated_report.md")
    alpha_available = bool(alpha) and np.isfinite(
        [
            alpha.get("corr_alpha_future_return", np.nan),
            alpha.get("corr_trade_alpha", np.nan),
            alpha.get("share_sign_trade_matches_alpha", np.nan),
            alpha.get("gross_alpha_capture", np.nan),
        ]
    ).all()

    rows = [
        ("Rolling pairs", f"{len(strategy_pairs)}"),
        ("Stock-days", f"{int(strategy['number_of_stock_days'].sum()):,}"),
        ("Mean daily P\\&L across pairs", f"\\${fmt_m(strategy['mean_daily_net_pnl'].mean())}m"),
        ("Average annualised Sharpe", f"{strategy['annualized_sharpe'].mean():.2f}"),
        ("Total net P\\&L", f"\\${fmt_m(fitted['total_net_pnl_fitted_proxy'].sum())}m"),
        ("Total local proxy cost", f"\\${fmt_m(fitted['total_fitted_proxy_cost'].sum())}m"),
        ("Total notional turnover", f"\\${fmt_bn(strategy['total_notional_turnover'].sum())}bn"),
        ("Worst max drawdown", fmt_signed_m(strategy["max_drawdown"].min())),
        (
            "Realised participation: max / mean",
            f"{fmt_part(strategy['max_participation_rate'].max())} / {fmt_part(strategy['mean_participation_rate'].mean())}",
        ),
        ("Maximum impact dislocation", f"{strategy['max_abs_impact'].max():.3f}"),
    ]

    if alpha_available:
        rows.extend(
            [
                (
                    "Pair 1 alpha correlations",
                    f"$\\rho(\\alpha,r_{{t+H}})={alpha['corr_alpha_future_return']:.3f}$; "
                    f"$\\rho(q,\\alpha)={alpha['corr_trade_alpha']:.3f}$",
                ),
                (
                    "Pair 1 sign alignment / gross alpha",
                    f"{100 * alpha['share_sign_trade_matches_alpha']:.1f}\\%; "
                    f"\\${fmt_m(alpha['gross_alpha_capture'])}m",
                ),
            ]
        )
        alpha_sentence = (
            "Pair 1 alpha diagnostics are included as representative row-level diagnostics; "
            "the retained all-pairs summaries do not store these diagnostics for every rolling pair."
        )
    else:
        rows.extend(
            [
                ("Pair 1 alpha correlations", "not available in retained all-pairs summary"),
                ("Pair 1 sign alignment / gross alpha", "not available in retained all-pairs summary"),
            ]
        )
        alpha_sentence = (
            "Alpha diagnostics are not available in the retained all-pairs summary, so they are not interpreted here."
        )

    if len(rows) > 12:
        fail(f"table has {len(rows)} rows; maximum is 12")

    table_path = OUT / "section_2_6_performance_metrics.tex"
    text_path = OUT / "section_2_6_text.tex"

    table_lines = [
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
    ]
    for metric, value in rows:
        table_lines.append(f"{metric} & {value} \\\\")
    table_lines.extend([r"\bottomrule", r"\end{tabular}"])
    table_path.write_text("\n".join(table_lines) + "\n")

    avg_daily = strategy["mean_daily_net_pnl"].mean()
    avg_sharpe = strategy["annualized_sharpe"].mean()
    total_net = fitted["total_net_pnl_fitted_proxy"].sum()
    text = (
        f"The table reports the out-of-sample performance "
        f"of the final reportable strategy, \\texttt{{OW\\_transient\\_proxy}}, over all rolling pairs. "
        f"The strategy remains profitable after local proxy costs, with mean daily P\\&L "
        f"of \\${fmt_m(avg_daily)}m, average annualised Sharpe of {avg_sharpe:.2f}, "
        f"and total net P\\&L of \\${fmt_m(total_net)}m across the all-pairs run. "
        f"{alpha_sentence}\n"
    )
    text_path.write_text(text)

    snippet = r"""\subsection{2.6 Performance Metrics}
\input{outputs/report_tables/section_2_6_text.tex}
\begin{table}[h]
\centering
\caption{Out-of-sample performance metrics for the final reportable strategy.}
\input{outputs/report_tables/section_2_6_performance_metrics.tex}
\end{table}"""

    print(table_path.relative_to(ROOT))
    print(text_path.relative_to(ROOT))
    print()
    print(snippet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
