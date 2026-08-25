"""
Backtest engine for the multi-timeframe signal trigger.

Walks historical 1h / 15m / 5m bars and asks: at each bar, would the
signal engine have fired BUY or SELL?  Then measures the forward return
of gold over a configurable holding window.

What it answers
---------------
* Hit rate (% of triggers whose forward return agreed with direction)
* Average R-multiple per trigger
* Expectancy (mean R)
* Profit factor
* Max drawdown
* Per-signal breakdown (BUY vs SELL hit rates)
* Threshold sensitivity table (how performance changes with the
  buy/sell signal thresholds)

Walk-forward rules (no look-ahead)
----------------------------------
* At each step T, we use bars up to T for the 5m / 15m / 1h lookbacks.
* The 1h signal is computed on the bar AT T, with its lookback window
  ending at T.
* We use the 1h bar at T as the "entry" and measure forward return
  on the 1h series from T to T+holding.
* This means we may execute the trade at the close of bar T and
  measure from the next bar onwards (T+1 close).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from .config import Config
    from .signals import (
        SignalConfig, SignalResult, signal_value, probability,
    )
    from .indicators import ema
except ImportError:
    from config import Config
    from signals import (
        SignalConfig, SignalResult, signal_value, probability,
    )
    from indicators import ema


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class SignalBacktestConfig:
    start: str = "2024-01-01"
    end:   str = "2030-12-31"
    holding_bars: int = 12          # forward window in 1h bars (= 0.5 trading day)
    cooldown_bars: int = 6          # minimum bars between successive triggers
    step_bars: int = 1              # evaluate every Nth 1h bar (1 = every bar)
    max_triggers: int = 5000        # safety cap

    # Optional threshold sweep (around the user's ±9 defaults)
    sweep_low:  float =  3.0
    sweep_high: float = 15.0
    sweep_step: float =  2.0


# -----------------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------------
@dataclass
class SignalBacktestResult:
    triggers: pd.DataFrame             # one row per BUY/SELL trigger
    equity_curve: pd.Series            # strategy equity (compounding)
    benchmark: pd.Series               # buy & hold gold equity over same window
    metrics: Dict[str, float]
    bucket_stats: pd.DataFrame         # per-trigger-type breakdown
    threshold_sweep: pd.DataFrame      # sensitivity table
    coverage_warning: Optional[str] = None   # shown if 5m/15m data is shorter than window


# -----------------------------------------------------------------------------
# Per-bar forecast label (the same logic as signals.forecast_signal_for_tf
# but operating on a slice that ends at a specific bar index)
# -----------------------------------------------------------------------------
def _label_at(close: pd.Series, periods: Dict[str, int], idx: int) -> tuple:
    """
    Return (label, score) computed on bars [:idx+1] only (no look-ahead).
    """
    window = close.iloc[: idx + 1]
    if len(window) < max(periods["ema_slow"], periods["rsi_len"]) + 5:
        return "Neutral", 0.0

    e_fast = ema(window, periods["ema_fast"]).iloc[-1]
    e_slow = ema(window, periods["ema_slow"]).iloc[-1]
    r_window = window.diff()
    gain = r_window.clip(lower=0.0)
    loss = (-r_window).clip(lower=0.0)
    avg_gain = gain.ewm(com=periods["rsi_len"] - 1, adjust=False,
                        min_periods=periods["rsi_len"]).mean().iloc[-1]
    avg_loss = loss.ewm(com=periods["rsi_len"] - 1, adjust=False,
                        min_periods=periods["rsi_len"]).mean().iloc[-1]
    if avg_loss == 0 or np.isnan(avg_loss):
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)

    t_score = 100.0 if e_fast > e_slow else (-100.0 if e_fast < e_slow else 0.0)
    pos = max(0.0, (rsi - 60.0) * 2.5)
    neg = -max(0.0, (40.0 - rsi) * 2.5)
    soft = (rsi - 50.0) * 0.5
    m_score = pos if rsi > 60 else (neg if rsi < 40 else soft)

    raw = t_score * 0.70 + m_score * 0.30
    raw = max(-100.0, min(100.0, raw))
    if   raw >  40: label = "Bullish"
    elif raw < -40: label = "Bearish"
    else:           label = "Neutral"
    return label, raw


def _signal_value_at(data_slice: Dict,
                     config: Config, idx: int) -> float:
    """Compute the per-market scores at bar `idx` and return signal value.

    `data_slice` may contain either DataFrames (with 'Close' column) or
    Series (already-extracted close prices).
    """
    try:
        from config import NEGATIVE_CORRELATIONS
    except ImportError:
        try:
            from .config import NEGATIVE_CORRELATIONS
        except ImportError:
            NEGATIVE_CORRELATIONS = {"dxy"}

    asset_scores: Dict[str, float] = {}
    for mkt, df in data_slice.items():
        if mkt not in config.weights:
            continue
        # Accept both DataFrame-with-Close and bare Series
        if isinstance(df, pd.Series):
            close = df
        elif isinstance(df, pd.DataFrame) and "Close" in df.columns:
            close = df["Close"]
        else:
            continue
        if len(close) < idx + 1 or idx < 0:
            continue
        if len(close.iloc[: idx + 1]) < config.periods["ema_slow"] + 5:
            continue
        lbl, sc = _label_at(close, config.periods, idx)
        v = sc
        if mkt in NEGATIVE_CORRELATIONS:
            v = -v
        asset_scores[mkt] = v
    return signal_value(asset_scores, config.weights), asset_scores


def _forecast_label_at(close: pd.Series, periods: Dict[str, int], idx: int) -> tuple:
    """Same as _label_at but using a custom ema/rsi set (for forecast horizons)."""
    return _label_at(close, periods, idx)


# -----------------------------------------------------------------------------
# Main backtest
# -----------------------------------------------------------------------------
def run_signal_backtest(
    data_5m:  Dict[str, pd.DataFrame],
    data_15m: Dict[str, pd.DataFrame],
    data_1h:  Dict[str, pd.DataFrame],
    config:   Config,
    sig_cfg:  SignalConfig = SignalConfig(),
    bt_cfg:   SignalBacktestConfig = SignalBacktestConfig(),
    initial_capital: float = 10_000.0,
) -> SignalBacktestResult:
    """
    Walk the 1h series, evaluate the multi-TF signal at each bar, and
    record triggers when BUY or SELL fires.
    """
    # ---- 1. Trim to backtest window ------------------------------------
    if "gold" not in data_1h or len(data_1h["gold"]) < 100:
        raise RuntimeError("Not enough 1h data to backtest.")

    gold_1h = data_1h["gold"]["Close"].copy()
    start_ts = pd.Timestamp(bt_cfg.start)
    end_ts   = pd.Timestamp(bt_cfg.end)
    if start_ts > gold_1h.index[-1] or end_ts < gold_1h.index[0]:
        raise RuntimeError(
            f"Backtest window {bt_cfg.start} → {bt_cfg.end} outside available data."
        )
    start_idx = max(gold_1h.index.get_indexer([start_ts], method="nearest")[0], 0)
    end_idx   = min(gold_1h.index.get_indexer([end_ts], method="nearest")[0], len(gold_1h) - 1)
    n_bars    = end_idx - start_idx + 1
    if n_bars < 50:
        raise RuntimeError(f"Backtest window too narrow ({n_bars} bars).")

    # ---- 1b. Validate 5m / 15m data coverage -------------------------
    # yfinance 5m has 60-day limit, 15m has 30-day limit.  Note in result
    # if the requested window extends past the available higher-TF data.
    coverage_warning = None
    if data_5m and "gold" in data_5m and len(data_5m["gold"]) > 0:
        first_5m = data_5m["gold"].index[0]
        if start_ts < first_5m:
            coverage_warning = (
                f"5m data only available from {first_5m.date()}; "
                f"earlier bars in window use only 15m + 1h. "
                f"yfinance 5m has a 60-day lookback limit."
            )
    if data_15m and "gold" in data_15m and len(data_15m["gold"]) > 0:
        first_15m = data_15m["gold"].index[0]
        if start_ts < first_15m and coverage_warning is None:
            coverage_warning = (
                f"15m data only available from {first_15m.date()}; "
                f"earlier bars use only 1h. yfinance 15m has a 30-day limit."
            )

    # ---- 2. Pre-compute per-bar 5m / 15m / 1h forecast labels -----------
    # For 5m and 15m we aggregate to the 1h clock so labels line up.
    def _resample_close(df_dict: Dict[str, pd.DataFrame], rule: str) -> Dict[str, pd.Series]:
        out: Dict[str, pd.Series] = {}
        for mkt, df in df_dict.items():
            if "Close" not in df.columns:
                continue
            s = df["Close"].resample(rule).last().dropna()
            out[mkt] = s
        return out

    if "gold" in data_5m:
        d5  = _resample_close(data_5m,  "1h")
        d15 = _resample_close(data_15m, "1h")
    else:
        d5  = {k: v["Close"] for k, v in data_5m.items()  if "Close" in v}
        d15 = {k: v["Close"] for k, v in data_15m.items() if "Close" in v}

    # ---- 3. Walk forward, evaluate signal ------------------------------
    triggers: List[Dict] = []
    last_trigger_idx = -10_000
    n_evaluated = 0

    for i in range(start_idx, end_idx - bt_cfg.holding_bars, bt_cfg.step_bars):
        n_evaluated += 1
        if n_evaluated > bt_cfg.max_triggers:
            break

        # Align 5m/15m to the current 1h timestamp
        ts = gold_1h.index[i]
        d5_i  = {mkt: s.loc[:ts] for mkt, s in d5.items()  if ts in s.index or s.index[-1] >= ts}
        d15_i = {mkt: s.loc[:ts] for mkt, s in d15.items() if ts in s.index or s.index[-1] >= ts}
        d1_i  = {mkt: df["Close"].iloc[: i + 1] for mkt, df in data_1h.items() if "Close" in df}

        # Forecast labels at this bar (no look-ahead)
        try:
            lbl_5m,  sc_5m  = _forecast_label_at(
                d5_i.get("gold", pd.Series(dtype=float)),
                {"ema_fast": config.forecasts["short"]["ema_fast"],
                 "ema_slow": config.forecasts["short"]["ema_slow"],
                 "rsi_len":  config.forecasts["short"]["rsi"]},
                len(d5_i.get("gold", pd.Series(dtype=float))) - 1,
            ) if len(d5_i.get("gold", pd.Series(dtype=float))) > 30 else ("Neutral", 0.0)
            lbl_15m, sc_15m = _forecast_label_at(
                d15_i.get("gold", pd.Series(dtype=float)),
                {"ema_fast": config.forecasts["medium"]["ema_fast"],
                 "ema_slow": config.forecasts["medium"]["ema_slow"],
                 "rsi_len":  config.forecasts["medium"]["rsi"]},
                len(d15_i.get("gold", pd.Series(dtype=float))) - 1,
            ) if len(d15_i.get("gold", pd.Series(dtype=float))) > 30 else ("Neutral", 0.0)
            lbl_1h,  sc_1h  = _forecast_label_at(
                d1_i.get("gold", pd.Series(dtype=float)),
                {"ema_fast": config.forecasts["long"]["ema_fast"],
                 "ema_slow": config.forecasts["long"]["ema_slow"],
                 "rsi_len":  config.forecasts["long"]["rsi"]},
                i,
            ) if len(d1_i.get("gold", pd.Series(dtype=float))) > 30 else ("Neutral", 0.0)
        except Exception:
            continue

        # Signal values
        sig_5m,  _ = _signal_value_at(d5_i,  config, len(d5_i.get("gold",  pd.Series(dtype=float))) - 1)
        sig_15m, _ = _signal_value_at(d15_i, config, len(d15_i.get("gold", pd.Series(dtype=float))) - 1)
        sig_1h,  _ = _signal_value_at(d1_i,  config, i)

        if not (np.isfinite(sig_5m) and np.isfinite(sig_15m) and np.isfinite(sig_1h)):
            continue

        # BUY condition
        momentum_ok_buy  = sig_15m > sig_1h
        buy_ok = (
            lbl_5m  == "Bullish" and
            lbl_15m == "Bullish" and
            lbl_1h  == "Bullish" and
            sig_1h  > sig_cfg.buy_1h_threshold and
            (momentum_ok_buy if sig_cfg.momentum_gap_required else True)
        )
        sell_ok = (
            lbl_5m  == "Bearish" and
            lbl_15m == "Bearish" and
            lbl_1h  == "Bearish" and
            sig_1h  < sig_cfg.sell_1h_threshold and
            ((not momentum_ok_buy) if sig_cfg.momentum_gap_required else True)
        )

        if not (buy_ok or sell_ok):
            continue
        if (i - last_trigger_idx) < bt_cfg.cooldown_bars:
            continue

        # Forward return over the holding window
        entry_price = float(gold_1h.iloc[i])
        exit_price  = float(gold_1h.iloc[i + bt_cfg.holding_bars])
        if entry_price <= 0 or not np.isfinite(exit_price):
            continue
        fwd_ret = (exit_price - entry_price) / entry_price
        side = 1 if buy_ok else -1
        signed_ret = side * fwd_ret

        # Probability estimate
        agree = (side == 1 and lbl_5m == "Bullish") or (side == -1 and lbl_5m == "Bearish")
        prob = probability(sig_1h, sig_15m, agree, sig_cfg)

        triggers.append({
            "entry_time":   gold_1h.index[i],
            "exit_time":    gold_1h.index[i + bt_cfg.holding_bars],
            "side":         "long" if side == 1 else "short",
            "signal":       "BUY" if buy_ok else "SELL",
            "entry":        entry_price,
            "exit":         exit_price,
            "fwd_return":   fwd_ret,
            "signed_return":signed_ret,
            "sig_5m":       sig_5m,
            "sig_15m":      sig_15m,
            "sig_1h":       sig_1h,
            "probability":  prob,
        })
        last_trigger_idx = i

    if not triggers:
        return SignalBacktestResult(
            triggers=pd.DataFrame(),
            equity_curve=pd.Series(dtype=float),
            benchmark=pd.Series(dtype=float),
            metrics={},
            bucket_stats=pd.DataFrame(),
            threshold_sweep=pd.DataFrame(),
            coverage_warning=coverage_warning,
        )

    trig_df = pd.DataFrame(triggers).set_index("entry_time").sort_index()

    # ---- 4. Equity curve -------------------------------------------------
    equity = pd.Series(index=gold_1h.index, dtype=float)
    equity.iloc[:] = np.nan
    for t in triggers:
        i_entry = gold_1h.index.get_loc(t["entry_time"])
        i_exit  = gold_1h.index.get_loc(t["exit_time"])
        # Mark-to-market between entry and exit linearly
        # (simple approximation: apply full return at exit bar)
        equity.iloc[i_exit] = t["signed_return"]

    # Compounding equity path
    bench_ret   = gold_1h.pct_change().fillna(0.0)
    strat_daily = bench_ret.copy()
    for t in triggers:
        i_entry = gold_1h.index.get_loc(t["entry_time"])
        i_exit  = gold_1h.index.get_loc(t["exit_time"])
        # Add the trigger's signed return to the corresponding exit bar
        strat_daily.iloc[i_exit] = (1 + strat_daily.iloc[i_exit]) * (1 + t["signed_return"]) - 1

    eq_curve = (1 + strat_daily).cumprod() * initial_capital
    bench    = (1 + bench_ret).cumprod() * initial_capital

    # ---- 5. Headline metrics --------------------------------------------
    metrics = _compute_metrics(trig_df, eq_curve, bench, initial_capital)

    # ---- 6. Per-bucket stats --------------------------------------------
    bucket_stats = _bucket_stats(trig_df)

    # ---- 7. Threshold sensitivity sweep ---------------------------------
    sweep = _threshold_sweep(
        trig_df, sig_cfg,
        low=bt_cfg.sweep_low, high=bt_cfg.sweep_high, step=bt_cfg.sweep_step,
    )

    return SignalBacktestResult(
        triggers    = trig_df,
        equity_curve= eq_curve,
        benchmark   = bench,
        metrics     = metrics,
        bucket_stats= bucket_stats,
        threshold_sweep = sweep,
        coverage_warning = coverage_warning,
    )


def _compute_metrics(
    trig_df: pd.DataFrame,
    eq: pd.Series,
    bench: pd.Series,
    initial_capital: float,
) -> Dict[str, float]:
    if trig_df.empty:
        return {
            "n_triggers": 0, "hit_rate_pct": 0.0, "expectancy": 0.0,
            "avg_R_pct": 0.0, "profit_factor": 0.0,
            "max_dd_pct": 0.0, "total_return_pct": 0.0,
            "annualized_pct": 0.0, "avg_holding_return_pct": 0.0,
        }

    r = trig_df["signed_return"]
    wins  = r[r > 0]
    losses= r[r < 0]
    gp = wins.sum()  if len(wins)  else 0.0
    gl = -losses.sum() if len(losses) else 0.0
    pf = float(gp / gl) if gl > 0 else float("inf")

    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = float(dd.min() * 100) if len(dd) else 0.0

    final = float(eq.iloc[-1]) if len(eq) else initial_capital
    total_ret = (final / initial_capital - 1) * 100
    days = (eq.index[-1] - eq.index[0]).days if len(eq) > 1 else 0
    ann = ((final / initial_capital) ** (365 / max(days, 1)) - 1) * 100 if days > 0 else 0.0

    return {
        "n_triggers":              int(len(trig_df)),
        "hit_rate_pct":            round(float((r > 0).mean() * 100), 2),
        "expectancy":              round(float(r.mean() * 100), 3),  # % per trade
        "avg_R_pct":               round(float(r.mean() * 100), 3),
        "profit_factor":           round(pf, 2) if np.isfinite(pf) else 99.99,
        "max_dd_pct":              round(max_dd, 2),
        "total_return_pct":        round(total_ret, 2),
        "annualized_pct":          round(ann, 2),
        "avg_holding_return_pct":  round(float(r.mean() * 100), 3),
    }


def _bucket_stats(trig_df: pd.DataFrame) -> pd.DataFrame:
    if trig_df.empty:
        return pd.DataFrame()
    g = trig_df.groupby("signal")["signed_return"]
    out = pd.DataFrame({
        "n":          g.count(),
        "hit_rate_%": (g.apply(lambda x: (x > 0).mean()) * 100).round(1),
        "avg_ret_%":  (g.mean() * 100).round(3),
        "total_ret_%":(g.sum() * 100).round(2),
        "max_win_%":  (g.max() * 100).round(2),
        "max_loss_%": (g.min() * 100).round(2),
    })
    return out.reindex([b for b in ("BUY", "SELL") if b in out.index])


def _threshold_sweep(
    trig_df: pd.DataFrame,
    base_cfg: SignalConfig,
    low: float, high: float, step: float,
) -> pd.DataFrame:
    """
    Re-evaluate each trigger under different signal thresholds without
    re-running the engine.  For each threshold T, count a trigger as
    valid only if abs(sig_1h) >= T.  This is a fast approximate.
    """
    if trig_df.empty:
        return pd.DataFrame()
    rows = []
    thresholds = np.arange(low, high + step / 2, step)
    for t in thresholds:
        df_filt = trig_df[trig_df["sig_1h"].abs() >= t]
        if df_filt.empty:
            rows.append({"threshold": round(t, 2), "n": 0, "hit_rate_%": 0.0,
                         "expectancy_%": 0.0})
            continue
        r = df_filt["signed_return"]
        rows.append({
            "threshold":    round(t, 2),
            "n":            int(len(df_filt)),
            "hit_rate_%":   round(float((r > 0).mean() * 100), 1),
            "expectancy_%": round(float(r.mean() * 100), 3),
        })
    return pd.DataFrame(rows)
