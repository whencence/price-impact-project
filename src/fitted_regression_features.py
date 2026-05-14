"""Feature engineering matching teammate's fitted regression notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd

META_COLUMNS = {"pair_id", "train_month", "test_month", "model", "stock", "half_life_sec", "source_file"}
OW_TRANSIENT_FEATURES = ["x_flow", "ow_state_pre"]


def make_seconds_from_open(df: pd.DataFrame) -> pd.DataFrame:
    """Add seconds_from_open from datetime/trading_date if missing."""

    out = df.copy()
    if "seconds_from_open" not in out.columns:
        open_ts = pd.to_datetime(out["trading_date"]) + pd.Timedelta(hours=9, minutes=30)
        out["seconds_from_open"] = (pd.to_datetime(out["datetime"]) - open_ts).dt.total_seconds()
    return out


def compute_training_scales(train_df: pd.DataFrame) -> dict[str, object]:
    """Compute stock-level flow/depth scales from train month only."""

    train = train_df.copy()
    train["abs_orderFlow"] = pd.to_numeric(train["orderFlow"], errors="coerce").abs().fillna(0.0)
    daily = train.groupby(["stock", "trading_date"], sort=False)["abs_orderFlow"].sum().reset_index()
    flow_scale = daily.groupby("stock")["abs_orderFlow"].median()
    depth_scale = pd.to_numeric(train["depth"], errors="coerce").groupby(train["stock"]).median()
    global_flow = float(np.nanmedian(flow_scale.to_numpy())) if len(flow_scale) else 1.0
    global_depth = float(np.nanmedian(depth_scale.to_numpy())) if len(depth_scale) else 1.0
    return {
        "flow_scale_by_stock": flow_scale.to_dict(),
        "depth_scale_by_stock": depth_scale.to_dict(),
        "global_flow_scale": max(global_flow, 1e-12),
        "global_depth_scale": max(global_depth, 1e-12),
    }


def _map_scale(df: pd.DataFrame, scales: dict[str, object], key: str, fallback_key: str) -> pd.Series:
    values = df["stock"].map(scales[key]).astype(float)
    return values.fillna(float(scales[fallback_key])).clip(lower=1e-12)


def prepare_model_frame(df: pd.DataFrame, scales: dict[str, object]) -> pd.DataFrame:
    """Prepare teammate regression target and features."""

    out = make_seconds_from_open(df.copy())
    for col in ["mid", "midEnd", "orderFlow", "trade", "hidden", "auction", "spread", "depth", "lobImb", "effLobImb"]:
        out[col] = pd.to_numeric(out.get(col, 0.0), errors="coerce")
    flow_scale = _map_scale(out, scales, "flow_scale_by_stock", "global_flow_scale")
    depth_scale = _map_scale(out, scales, "depth_scale_by_stock", "global_depth_scale")
    depth = out["depth"].replace(0, np.nan).fillna(depth_scale).clip(lower=1e-12)
    out["ret_bps"] = 10000.0 * (out["midEnd"] - out["mid"]) / out["mid"]
    out["flow_scale"] = flow_scale
    out["depth_scale"] = depth_scale
    out["x_flow"] = out["orderFlow"] / flow_scale
    out["x_trade"] = out["trade"] / flow_scale
    out["x_hidden"] = out["hidden"] / flow_scale
    out["x_auction"] = out["auction"] / flow_scale
    out["spread_bps"] = 10000.0 * out["spread"] / out["mid"]
    out["x_flow_depth"] = out["orderFlow"] / depth
    out["x_trade_depth"] = out["trade"] / depth
    fill_cols = ["x_flow", "x_trade", "x_hidden", "x_auction", "spread_bps", "x_flow_depth", "x_trade_depth", "lobImb", "effLobImb", "depth"]
    out[fill_cols] = out[fill_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = out[np.isfinite(pd.to_numeric(out["ret_bps"], errors="coerce"))].copy()
    return out.sort_values(["stock", "trading_date", "datetime"]).reset_index(drop=True)


def add_ow_state_one_stock(df: pd.DataFrame, half_life_sec: float, flow_col: str = "x_flow") -> pd.DataFrame:
    """Add teammate OW state using exp(-dt / half_life_sec), with daily reset."""

    out = df.copy().sort_values(["trading_date", "datetime"]).reset_index(drop=True)
    pre_values, post_values = [], []
    prev_state = 0.0
    prev_day = None
    prev_time = None
    tau = max(float(half_life_sec), 1e-12)
    for _, row in out.iterrows():
        day = row["trading_date"]
        ts = pd.to_datetime(row["datetime"])
        if prev_day != day:
            prev_state = 0.0
            dt = 0.0
        else:
            dt = max((ts - prev_time).total_seconds(), 0.0)
        phi = float(np.exp(-dt / tau))
        pre = phi * prev_state
        post = pre + float(row.get(flow_col, 0.0))
        pre_values.append(pre)
        post_values.append(post)
        prev_state = post
        prev_day = day
        prev_time = ts
    out["ow_state_pre"] = pre_values
    out["ow_state_post"] = post_values
    return out


def add_ow_state_by_stock(df: pd.DataFrame, params_by_stock: dict[str, float]) -> pd.DataFrame:
    """Add OW state per stock using stock-specific teammate time constant."""

    pieces = []
    for stock, group in df.groupby("stock", sort=False):
        pieces.append(add_ow_state_one_stock(group, params_by_stock.get(str(stock), 60.0)))
    return pd.concat(pieces, ignore_index=True).sort_values(["stock", "trading_date", "datetime"]).reset_index(drop=True)


def infer_feature_columns_from_params(params_df: pd.DataFrame) -> list[str]:
    """Infer numeric coefficient columns from teammate params."""

    cols = []
    for col in params_df.columns:
        if col in META_COLUMNS or col == "intercept":
            continue
        if pd.api.types.is_numeric_dtype(params_df[col]) and params_df[col].notna().any():
            cols.append(col)
    return cols
