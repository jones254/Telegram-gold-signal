"""
Multi-timeframe signal engine for the Gold Scalper.

Computes the BUYSIGNAL / SELLSIGNAL / WAITBUY / WAITSELL / NOACTION
triggers based on the user's spec:

  signal(tf) = (((DXY + VIX) * -1) + SP500) * 2
  where each score is the asset's per-market score from scoring.py
  (post sign-flip for DXY, since DXY is in NEGATIVE_CORRELATIONS).

  BUY  : 5m forecast bullish (short+medium) AND
         15m forecast bullish (short AND medium) AND
         15m signal > 1h signal  (momentum building, not fading) AND
         1h forecast bullish (short) AND 1h signal > 9

  SELL : mirror of BUY with Bearish and thresholds inverted

  WAITBUY  : 15m signal > 3 AND 1h signal > 3  (no full trigger, but leaning)
  WAITSELL : 15m signal < -3 AND 1h signal < -3
  NOACTION : anything else

Probability % is a logistic transform of:
   - the magnitude of 1h signal
   - the agreement between 15m and 1h (smaller gap = higher confidence)
   - whether 5m agrees with the direction

Entry snapshot (render_chart_snapshot) renders a small Plotly chart
to PNG bytes for use as a Telegram photo attachment.
"""

from __future__ import annotations
import io
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    from .scoring import asset_score
    from .indicators import ema
except ImportError:
    from scoring import asset_score
    from indicators import ema


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class SignalConfig:
    # 5m: only short and medium forecasts are reliable (long EMAs need 200 bars = 17h)
    require_5m_short: bool = True
    require_5m_medium: bool = True
    require_5m_long: bool = False   # usually disabled; see above

    # 15m: short AND medium must agree
    require_15m_short: bool = True
    require_15m_medium: bool = True
    require_15m_long: bool = False

    # 1h: only short required (the 1h signal value does the heavy lifting)
    require_1h_short: bool = True
    require_1h_long: bool = False

    # Buy/sell thresholds on the signal value
    buy_1h_threshold: float = 9.0
    sell_1h_threshold: float = -9.0

    # Stand-down thresholds
    wait_buy_threshold: float = 3.0
    wait_sell_threshold: float = -3.0

    # Momentum filter: how much bigger 15m must be vs 1h for the trigger
    momentum_gap_required: bool = True

    # Probability scaling
    prob_min: float = 0.50
    prob_max: float = 0.90


# -----------------------------------------------------------------------------
# State container — what's the latest signal + history
# -----------------------------------------------------------------------------
@dataclass
class SignalState:
    last_signal: str = "NOACTION"   # BUY / SELL / WAITBUY / WAITSELL / NOACTION
    last_probability: float = 0.0
    last_check_ts: pd.Timestamp = None
    last_1h_signal: float = 0.0
    last_15m_signal: float = 0.0
    last_5m_signal: float = 0.0
    last_5m_label: str = "Neutral"
    last_15m_label: str = "Neutral"
    last_1h_label: str = "Neutral"
    last_price: float = 0.0
    history: List[Dict] = field(default_factory=list)

    def add_history(self, entry: Dict, maxlen: int = 200) -> None:
        self.history.insert(0, entry)
        if len(self.history) > maxlen:
            self.history = self.history[:maxlen]


# -----------------------------------------------------------------------------
# Per-timeframe engine: produce a (label, score) pair for any interval
# -----------------------------------------------------------------------------
def forecast_signal_for_tf(
    data: Dict[str, pd.DataFrame],
    config,
    ema_fast_key: str = "short",
) -> Tuple[str, float, Dict[str, float]]:
    """
    Compute the forecast label + score for a specific data slice.

    Returns (label, score, asset_scores) where asset_scores is the
    per-market score dict for downstream signal calculation.
    """
    ema_cfg = config.forecasts[ema_fast_key]
    close = data["gold"]["Close"]
    fs = asset_score(close, {
        "ema_fast": ema_cfg["ema_fast"],
        "ema_slow": ema_cfg["ema_slow"],
        "rsi_len":  ema_cfg["rsi"],
        "roc_len":  config.periods["roc_len"],
    })
    score = float(fs.iloc[-1]) if not pd.isna(fs.iloc[-1]) else 0.0

    if   score >  40: label = "Bullish"
    elif score < -40: label = "Bearish"
    else:             label = "Neutral"

    # Also compute the multi-market asset scores for the signal formula
    asset_scores = _compute_asset_scores(data, config)
    return label, score, asset_scores


def _compute_asset_scores(data: Dict[str, pd.DataFrame], config) -> Dict[str, float]:
    """Per-market asset score on the latest bar, post sign-flip."""
    try:
        from .config import NEGATIVE_CORRELATIONS
    except ImportError:
        from config import NEGATIVE_CORRELATIONS

    scores: Dict[str, float] = {}
    for mkt, df in data.items():
        if mkt not in config.weights or "Close" not in df.columns or len(df) < 30:
            continue
        s = asset_score(df["Close"], config.periods)
        v = float(s.iloc[-1]) if not pd.isna(s.iloc[-1]) else 0.0
        if mkt in NEGATIVE_CORRELATIONS:
            v = -v
        scores[mkt] = v
    return scores


# -----------------------------------------------------------------------------
# The signal value the user specified
# -----------------------------------------------------------------------------
def signal_value(asset_scores: Dict[str, float]) -> float:
    """
    signal = (((DXY + VIX) * -1) + SP500) * 2

    Uses post-flip DXY (so a positive DXY-score here means dollar is
    strong, which is gold-hostile).  Higher value = more gold-friendly.
    """
    dxy = asset_scores.get("dxy", 0.0)
    vix = asset_scores.get("vix", 0.0)
    sp  = asset_scores.get("sp500", 0.0)
    return ((dxy + vix) * -1.0 + sp) * 2.0


# -----------------------------------------------------------------------------
# Probability estimate
# -----------------------------------------------------------------------------
def probability(signal_1h: float, signal_15m: float, agrees_5m: bool,
                cfg: SignalConfig) -> float:
    """
    Logistic-style confidence in the range [prob_min, prob_max].

    Inputs:
      signal_1h  : the 1h signal value (already weighted by the user's *2)
      signal_15m : the 15m signal value
      agrees_5m  : whether 5m forecast agrees with the direction
    """
    magnitude = min(abs(signal_1h), 18.0) / 18.0          # 0..1
    gap = abs(signal_1h - signal_15m)
    agreement = max(0.0, 1.0 - gap / 12.0)                # 0..1
    five_m_bonus = 0.10 if agrees_5m else 0.0

    score = 0.55 * magnitude + 0.35 * agreement + 0.10 + five_m_bonus
    score = max(0.0, min(1.0, score))
    return cfg.prob_min + score * (cfg.prob_max - cfg.prob_min)


# -----------------------------------------------------------------------------
# Main trigger check
# -----------------------------------------------------------------------------
@dataclass
class SignalResult:
    signal: str                  # BUY / SELL / WAITBUY / WAITSELL / NOACTION
    probability: float
    signal_5m: float
    signal_15m: float
    signal_1h: float
    label_5m: str
    label_15m: str
    label_1h: str
    last_price: float
    momentum_ok: bool
    agreement_5m: bool
    reasons: List[str]           # human-readable explanation


def evaluate(
    data_5m:  Dict[str, pd.DataFrame],
    data_15m: Dict[str, pd.DataFrame],
    data_1h:  Dict[str, pd.DataFrame],
    config,
    cfg: SignalConfig = SignalConfig(),
) -> SignalResult:
    """Run the full multi-timeframe check and return the trigger result."""
    # Compute labels + scores per timeframe
    label_5m,  score_5m,  as_5m  = forecast_signal_for_tf(data_5m,  config, "short")
    label_15m, score_15m, as_15m = forecast_signal_for_tf(data_15m, config, "medium")
    label_1h,  score_1h,  as_1h  = forecast_signal_for_tf(data_1h,  config, "long")

    sig_5m  = signal_value(as_5m)
    sig_15m = signal_value(as_15m)
    sig_1h  = signal_value(as_1h)

    last_price = float(data_1h["gold"]["Close"].iloc[-1])

    # ---- Determine agreement & momentum --------------------------------
    agree_5m_long = (label_5m == "Bullish")
    agree_5m_short = (label_5m == "Bearish")
    agree_5m = agree_5m_long or agree_5m_short  # any directional read on 5m

    momentum_ok_buy  = (sig_15m > sig_1h)
    momentum_ok_sell = (sig_15m < sig_1h)

    reasons: List[str] = []

    # ---- BUY check ------------------------------------------------------
    buy_ok = True
    if cfg.require_5m_short  and label_5m  != "Bullish":  buy_ok = False; reasons.append("5m short not bullish")
    if cfg.require_5m_medium and label_5m  != "Bullish":  buy_ok = False; reasons.append("5m medium not bullish")
    if cfg.require_5m_long   and label_5m  != "Bullish":  buy_ok = False; reasons.append("5m long not bullish")
    if cfg.require_15m_short and label_15m != "Bullish":  buy_ok = False; reasons.append("15m short not bullish")
    if cfg.require_15m_medium and label_15m != "Bullish": buy_ok = False; reasons.append("15m medium not bullish")
    if cfg.require_15m_long  and label_15m != "Bullish":  buy_ok = False; reasons.append("15m long not bullish")
    if cfg.require_1h_short  and label_1h  != "Bullish":  buy_ok = False; reasons.append("1h short not bullish")
    if sig_1h <= cfg.buy_1h_threshold:                    buy_ok = False; reasons.append(f"1h signal {sig_1h:.1f} <= {cfg.buy_1h_threshold}")
    if cfg.momentum_gap_required and not momentum_ok_buy: buy_ok = False; reasons.append("15m signal not building vs 1h")

    if buy_ok:
        return SignalResult(
            signal="BUY",
            probability=probability(sig_1h, sig_15m, agree_5m_long, cfg),
            signal_5m=sig_5m, signal_15m=sig_15m, signal_1h=sig_1h,
            label_5m=label_5m, label_15m=label_15m, label_1h=label_1h,
            last_price=last_price, momentum_ok=True, agreement_5m=agree_5m_long,
            reasons=["all BUY conditions met"],
        )

    # ---- SELL check -----------------------------------------------------
    sell_ok = True
    if cfg.require_5m_short  and label_5m  != "Bearish":  sell_ok = False; reasons.append("5m short not bearish")
    if cfg.require_5m_medium and label_5m  != "Bearish":  sell_ok = False; reasons.append("5m medium not bearish")
    if cfg.require_5m_long   and label_5m  != "Bearish":  sell_ok = False; reasons.append("5m long not bearish")
    if cfg.require_15m_short and label_15m != "Bearish":  sell_ok = False; reasons.append("15m short not bearish")
    if cfg.require_15m_medium and label_15m != "Bearish": sell_ok = False; reasons.append("15m medium not bearish")
    if cfg.require_15m_long  and label_15m != "Bearish":  sell_ok = False; reasons.append("15m long not bearish")
    if cfg.require_1h_short  and label_1h  != "Bearish":  sell_ok = False; reasons.append("1h short not bearish")
    if sig_1h >= cfg.sell_1h_threshold:                    sell_ok = False; reasons.append(f"1h signal {sig_1h:.1f} >= {cfg.sell_1h_threshold}")
    if cfg.momentum_gap_required and not momentum_ok_sell: sell_ok = False; reasons.append("15m signal not building vs 1h")

    if sell_ok:
        return SignalResult(
            signal="SELL",
            probability=probability(sig_1h, sig_15m, agree_5m_short, cfg),
            signal_5m=sig_5m, signal_15m=sig_15m, signal_1h=sig_1h,
            label_5m=label_5m, label_15m=label_15m, label_1h=label_1h,
            last_price=last_price, momentum_ok=True, agreement_5m=agree_5m_short,
            reasons=["all SELL conditions met"],
        )

    # ---- WAITBUY / WAITSELL --------------------------------------------
    if sig_15m > cfg.wait_buy_threshold and sig_1h > cfg.wait_buy_threshold:
        return SignalResult(
            signal="WAITBUY", probability=probability(sig_1h, sig_15m, agree_5m_long, cfg),
            signal_5m=sig_5m, signal_15m=sig_15m, signal_1h=sig_1h,
            label_5m=label_5m, label_15m=label_15m, label_1h=label_1h,
            last_price=last_price, momentum_ok=momentum_ok_buy, agreement_5m=agree_5m_long,
            reasons=[f"15m signal {sig_15m:.1f} and 1h signal {sig_1h:.1f} both > {cfg.wait_buy_threshold}"],
        )
    if sig_15m < cfg.wait_sell_threshold and sig_1h < cfg.wait_sell_threshold:
        return SignalResult(
            signal="WAITSELL", probability=probability(sig_1h, sig_15m, agree_5m_short, cfg),
            signal_5m=sig_5m, signal_15m=sig_15m, signal_1h=sig_1h,
            label_5m=label_5m, label_15m=label_15m, label_1h=label_1h,
            last_price=last_price, momentum_ok=momentum_ok_sell, agreement_5m=agree_5m_short,
            reasons=[f"15m signal {sig_15m:.1f} and 1h signal {sig_1h:.1f} both < {cfg.wait_sell_threshold}"],
        )

    return SignalResult(
        signal="NOACTION", probability=0.0,
        signal_5m=sig_5m, signal_15m=sig_15m, signal_1h=sig_1h,
        label_5m=label_5m, label_15m=label_15m, label_1h=label_1h,
        last_price=last_price, momentum_ok=False, agreement_5m=False,
        reasons=["no condition met"],
    )


# -----------------------------------------------------------------------------
# Telegram message formatting
# -----------------------------------------------------------------------------
def format_signal_message(res: SignalResult) -> Tuple[str, str]:
    """Return (caption, parse_mode) for the Telegram message."""
    if res.signal == "BUY":
        header = f"🟢 *BUY SIGNAL* — Gold (prob {res.probability*100:.0f}%)"
    elif res.signal == "SELL":
        header = f"🔴 *SELL SIGNAL* — Gold (prob {res.probability*100:.0f}%)"
    elif res.signal == "WAITBUY":
        header = f"⏳ *Waiting for BUY* — bias {res.probability*100:.0f}%"
    elif res.signal == "WAITSELL":
        header = f"⏳ *Waiting for SELL* — bias {res.probability*100:.0f}%"
    else:
        header = "💤 *No action*"

    body = (
        f"\n\n"
        f"• Last: `{res.last_price:,.2f}`\n"
        f"• 5m  forecast: *{res.label_5m}*  (signal `{res.signal_5m:+.2f}`)\n"
        f"• 15m forecast: *{res.label_15m}*  (signal `{res.signal_15m:+.2f}`)\n"
        f"• 1h  forecast: *{res.label_1h}*  (signal `{res.signal_1h:+.2f}`)\n"
        f"• Momentum: {'building' if res.momentum_ok else 'fading'}\n"
    )
    if res.reasons and res.signal in ("NOACTION", "WAITBUY", "WAITSELL"):
        body += f"\n_Reason:_ {', '.join(res.reasons)}"
    return header + body, "Markdown"


# -----------------------------------------------------------------------------
# Chart snapshot — returns PNG bytes for Telegram photo upload
# -----------------------------------------------------------------------------
def render_chart_snapshot(
    data_1h: Dict[str, pd.DataFrame],
    data_15m: Dict[str, pd.DataFrame],
    res: SignalResult,
    last_n_bars_1h: int = 60,
    last_n_bars_15m: int = 96,
) -> Optional[bytes]:
    """
    Render a small 2-panel chart: 1h gold + 15m gold, with the signal
    value annotated.  Returns PNG bytes suitable for Telegram sendPhoto.

    Requires `kaleido` to be installed; if not, returns None.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    gold_1h  = data_1h["gold"]["Close"].tail(last_n_bars_1h)
    gold_15m = data_15m["gold"]["Close"].tail(last_n_bars_15m)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.10,
        subplot_titles=("Gold — 1h", "Gold — 15m"),
    )
    fig.add_trace(go.Scatter(x=gold_1h.index, y=gold_1h.values, name="1h",
                             line=dict(color="#DAA520", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=gold_15m.index, y=gold_15m.values, name="15m",
                             line=dict(color="#888", width=1.5)), row=2, col=1)

    color = {"BUY": "#2E8B57", "SELL": "#B22222", "WAITBUY": "#90EE90",
             "WAITSELL": "#FFA07A", "NOACTION": "#888"}.get(res.signal, "#888")
    fig.add_annotation(
        text=f"{res.signal} · p={res.probability*100:.0f}%",
        xref="paper", yref="paper", x=0.99, y=1.06,
        showarrow=False, font=dict(color=color, size=14),
    )
    fig.update_layout(
        height=500, width=900, template="plotly_white",
        showlegend=False, margin=dict(l=10, r=10, t=40, b=10),
    )

    try:
        return fig.to_image(format="png", engine="kaleido")
    except Exception:
        return None
