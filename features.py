"""
Feature Engineering — Lean core set + "Expert" (EA-style) indicators.

Core set (18/TF):  returns, EMA trend, RSI, MACD, Bollinger, ATR, Stochastic,
                    ADX, CVD, VWAP, Donchian, S/R distance, volume, candle body.

Expert set (11/TF): Supertrend, Ichimoku, Parabolic SAR, Market Structure (BOS),
                    Fibonacci proximity, MA Ribbon alignment, Choppiness Index,
                    and an "EA Consensus" vote combining the trend-following experts.

Total: 29 features/TF × 3 TFs = 87, + 3 session features = 90 total.
"""

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════
#   CORE indicators
# ════════════════════════════════════════════════════════════════

def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return (100 - 100 / (1 + gain / (loss + 1e-10))) / 100

def atr(high, low, close, period=14):
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(high, low, close, period=14):
    up   = high.diff().clip(lower=0)
    down = (-low.diff()).clip(lower=0)
    pdm  = up.where(up > down, 0.0)
    ndm  = down.where(down > up, 0.0)
    atr_ = atr(high, low, close, period)
    pdi  = pdm.rolling(period).mean() / (atr_ + 1e-10)
    ndi  = ndm.rolling(period).mean() / (atr_ + 1e-10)
    dx   = (pdi - ndi).abs() / (pdi + ndi + 1e-10)
    return dx.rolling(period).mean()

def vwap_daily(high, low, close, volume):
    typical = (high + low + close) / 3
    df_tmp = pd.DataFrame({"tp": typical, "vol": volume,
                           "tpvol": typical * volume,
                           "tp2vol": (typical ** 2) * volume})
    results = {"vwap": [], "u1": [], "l1": []}
    for _, grp in df_tmp.groupby(df_tmp.index.date):
        cum_tpvol  = grp["tpvol"].cumsum()
        cum_tp2vol = grp["tp2vol"].cumsum()
        cum_vol    = grp["vol"].cumsum()
        vw = cum_tpvol / (cum_vol + 1e-10)
        var = (cum_tp2vol / (cum_vol + 1e-10) - vw ** 2).clip(lower=0)
        std = np.sqrt(var)
        for idx in grp.index:
            i = grp.index.get_loc(idx)
            results["vwap"].append((idx, vw.iloc[i]))
            results["u1"].append((idx, vw.iloc[i] + std.iloc[i]))
            results["l1"].append((idx, vw.iloc[i] - std.iloc[i]))
    def to_series(pairs):
        idx, vals = zip(*pairs)
        return pd.Series(vals, index=idx)
    return to_series(results["vwap"]), to_series(results["u1"]), to_series(results["l1"])

def cvd(close, open_, volume, period=20):
    delta    = volume * np.sign(close - open_)
    cvd_raw  = delta.rolling(period).sum()
    avg_vol  = volume.rolling(period).mean() * period
    cvd_norm = cvd_raw / (avg_vol + 1e-10)
    return cvd_norm, cvd_norm.diff(5)

def donchian_breakout(high, low, close, period=20):
    don_high = high.rolling(period).max()
    don_low  = low.rolling(period).min()
    rng      = don_high - don_low + 1e-10
    position = (close - don_low) / rng
    breakout = pd.Series(0.0, index=close.index)
    breakout[close >= don_high.shift(1)] =  1.0
    breakout[close <= don_low.shift(1)]  = -1.0
    return position, breakout

def support_resistance(high, low, close, period=10, lookback=100):
    swing_high = high.rolling(period, center=True).max() == high
    swing_low  = low.rolling(period,  center=True).min() == low
    res_dist = pd.Series(index=close.index, dtype=float)
    sup_dist = pd.Series(index=close.index, dtype=float)
    atr14 = atr(high, low, close, 14)
    for i in range(lookback, len(close)):
        cur, av = close.iloc[i], atr14.iloc[i]
        highs_above = high[swing_high].iloc[max(0, i-lookback):i]
        highs_above = highs_above[highs_above > cur]
        res = (highs_above.min() - cur) / (av + 1e-10) if len(highs_above) else 5.0
        lows_below = low[swing_low].iloc[max(0, i-lookback):i]
        lows_below = lows_below[lows_below < cur]
        sup = (cur - lows_below.max()) / (av + 1e-10) if len(lows_below) else 5.0
        res_dist.iloc[i] = min(res, 5.0)
        sup_dist.iloc[i] = min(sup, 5.0)
    return res_dist.fillna(5.0), sup_dist.fillna(5.0)


# ════════════════════════════════════════════════════════════════
#   "EXPERT" indicators — each is a mini rule-based EA
# ════════════════════════════════════════════════════════════════

def supertrend(high, low, close, period=10, multiplier=3.0):
    """Classic trend-following EA indicator. Returns (direction -1/+1, distance/ATR)."""
    atr_val = atr(high, low, close, period)
    hl2 = ((high + low) / 2).values
    basic_upper = hl2 + multiplier * atr_val.values
    basic_lower = hl2 - multiplier * atr_val.values
    c = close.values
    n = len(c)

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    direction = np.ones(n)
    st = np.zeros(n)

    for i in range(1, n):
        final_upper[i] = basic_upper[i] if (basic_upper[i] < final_upper[i-1] or
                                            c[i-1] > final_upper[i-1]) else final_upper[i-1]
        final_lower[i] = basic_lower[i] if (basic_lower[i] > final_lower[i-1] or
                                            c[i-1] < final_lower[i-1]) else final_lower[i-1]
        if direction[i-1] == 1:
            direction[i] = -1 if c[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if c[i] > final_upper[i] else -1
        st[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    dist = (c - st) / (atr_val.values + 1e-10)
    return (pd.Series(direction, index=close.index),
            pd.Series(np.clip(dist, -5, 5), index=close.index))


def parabolic_sar(high, low, close, step=0.02, max_step=0.2):
    """Classic trend-reversal EA indicator. Returns (direction -1/+1, distance/ATR)."""
    h, l, c = high.values, low.values, close.values
    n = len(c)
    sar = np.zeros(n)
    direction = np.ones(n)
    ep = h[0]
    af = step
    sar[0] = l[0]

    for i in range(1, n):
        prev_sar = sar[i-1]
        if direction[i-1] == 1:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], l[i-1], l[i-2] if i > 1 else l[i-1])
            if l[i] < sar[i]:
                direction[i], sar[i], ep, af = -1, ep, l[i], step
            else:
                direction[i] = 1
                if h[i] > ep:
                    ep, af = h[i], min(af + step, max_step)
        else:
            sar[i] = prev_sar - af * (prev_sar - ep)
            sar[i] = max(sar[i], h[i-1], h[i-2] if i > 1 else h[i-1])
            if h[i] > sar[i]:
                direction[i], sar[i], ep, af = 1, ep, h[i], step
            else:
                direction[i] = -1
                if l[i] < ep:
                    ep, af = l[i], min(af + step, max_step)

    atr_val = atr(high, low, close, 14).values
    dist = (c - sar) / (atr_val + 1e-10)
    return (pd.Series(direction, index=close.index),
            pd.Series(np.clip(dist, -5, 5), index=close.index))


def ichimoku_features(high, low, close, conv=9, base=26, span_b=52):
    """Cloud position (-1 below / 0 inside / +1 above) + Tenkan/Kijun cross sign."""
    tenkan = (high.rolling(conv).max() + low.rolling(conv).min()) / 2
    kijun  = (high.rolling(base).max() + low.rolling(base).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high.rolling(span_b).max() + low.rolling(span_b).min()) / 2

    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

    position = pd.Series(0.0, index=close.index)
    position[close > cloud_top] = 1.0
    position[close < cloud_bot] = -1.0
    cross = np.sign(tenkan - kijun).fillna(0)
    return position, cross


def market_structure_bos(high, low, close, period=10):
    """Break Of Structure — ICT/SMC style. +1 bullish BOS, -1 bearish BOS, 0 none."""
    swing_high = (high.rolling(period, center=True).max() == high).values
    swing_low  = (low.rolling(period,  center=True).min() == low).values
    h, l, c = high.values, low.values, close.values
    n = len(c)
    bos = np.zeros(n)
    last_high, last_low = np.nan, np.nan

    for i in range(n):
        if swing_high[i]:
            last_high = h[i]
        if swing_low[i]:
            last_low = l[i]
        if not np.isnan(last_high) and c[i] > last_high:
            bos[i] = 1.0
            last_high = np.nan
        elif not np.isnan(last_low) and c[i] < last_low:
            bos[i] = -1.0
            last_low = np.nan

    return pd.Series(bos, index=close.index)


def fibonacci_proximity(high, low, close, lookback=100):
    """Distance (0=at a key fib level, 1=far) to the nearest 38.2/50/61.8% retracement."""
    h, l, c = high.values, low.values, close.values
    atr_val = atr(high, low, close, 14).values
    n = len(c)
    prox = np.full(n, 5.0)

    for i in range(lookback, n):
        swing_h = h[i-lookback:i].max()
        swing_l = l[i-lookback:i].min()
        rng = swing_h - swing_l
        if rng <= 0:
            continue
        levels = (swing_l + rng*0.382, swing_l + rng*0.5, swing_l + rng*0.618)
        prox[i] = min(abs(c[i] - lvl) for lvl in levels) / (atr_val[i] + 1e-10)

    return pd.Series(np.clip(prox, 0, 5) / 5, index=close.index)


def ma_ribbon_score(close, spans=(5, 10, 20, 50, 100)):
    """+1 = all EMAs perfectly bullish-aligned, -1 = perfectly bearish-aligned."""
    emas = [close.ewm(span=s).mean() for s in spans]
    score = pd.Series(0.0, index=close.index)
    pairs = len(spans) - 1
    for i in range(pairs):
        score += np.sign(emas[i] - emas[i+1])
    return (score / pairs).fillna(0)


def choppiness_index(high, low, close, period=14):
    """~0 = strongly trending, ~1 = choppy/ranging. Complements ADX."""
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr_sum = tr.rolling(period).sum()
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    chop = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
    return (chop / 100).clip(0, 1.5).fillna(0.5)


# ════════════════════════════════════════════════════════════════
#   Per-timeframe feature assembly (29 features)
# ════════════════════════════════════════════════════════════════

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

    # ── Core (18) ──────────────────────────────────────────────
    f["ret5"]      = c.pct_change(5)
    f["ema21"]     = c.ewm(span=21).mean() / c - 1
    f["ema50"]     = c.ewm(span=50).mean() / c - 1
    f["rsi14"]     = rsi(c, 14)
    m = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    f["macd_hist"] = (m - m.ewm(span=9).mean()) / c
    bm, bsd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pct"]    = (c - (bm - 2*bsd)) / (4*bsd + 1e-10)
    atr14 = atr(h, l, c, 14)
    f["atr14"]     = atr14 / c
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    f["stoch_k"]   = (c - lo14) / (hi14 - lo14 + 1e-10)
    f["adx14"]     = adx(h, l, c, 14)
    cvd_v, cvd_s = cvd(c, o, v, 20)
    f["cvd"]       = cvd_v.clip(-3, 3)
    f["cvd_slope"] = cvd_s.clip(-1, 1)
    try:
        vw, _, _ = vwap_daily(h, l, c, v)
        f["vwap_dist"] = ((c - vw) / (atr14 + 1e-10)).clip(-5, 5)
    except Exception:
        f["vwap_dist"] = 0.0
    don_pos, don_brk = donchian_breakout(h, l, c, 20)
    f["don_pos"]   = don_pos
    f["don_brk"]   = don_brk
    try:
        res_d, sup_d = support_resistance(h, l, c, 10, 100)
        f["res_dist"] = (res_d / 5).clip(0, 1)
        f["sup_dist"] = (sup_d / 5).clip(0, 1)
    except Exception:
        f["res_dist"] = 0.0; f["sup_dist"] = 0.0
    f["vol_ratio"] = v / (v.rolling(20).mean() + 1e-10)
    f["body"]      = (c - o) / (h - l + 1e-10)

    # ── Expert / EA-style (11) ────────────────────────────────
    try:
        st_dir, st_dist = supertrend(h, l, c, 10, 3.0)
        f["st_dir"]  = st_dir
        f["st_dist"] = st_dist
    except Exception:
        f["st_dir"] = 0.0; f["st_dist"] = 0.0

    try:
        psar_dir, psar_dist = parabolic_sar(h, l, c)
        f["psar_dir"]  = psar_dir
        f["psar_dist"] = psar_dist
    except Exception:
        f["psar_dir"] = 0.0; f["psar_dist"] = 0.0

    try:
        ichi_pos, ichi_cross = ichimoku_features(h, l, c)
        f["ichi_pos"]   = ichi_pos
        f["ichi_cross"] = ichi_cross
    except Exception:
        f["ichi_pos"] = 0.0; f["ichi_cross"] = 0.0

    try:
        f["bos"] = market_structure_bos(h, l, c, 10)
    except Exception:
        f["bos"] = 0.0

    try:
        f["fib_prox"] = fibonacci_proximity(h, l, c, 100)
    except Exception:
        f["fib_prox"] = 1.0

    f["ma_ribbon"] = ma_ribbon_score(c)

    try:
        f["chop"] = choppiness_index(h, l, c, 14)
    except Exception:
        f["chop"] = 0.5

    # "EA Consensus" — average vote of the 4 trend-following experts above
    f["ea_consensus"] = (f["st_dir"].fillna(0) + f["ichi_pos"].fillna(0) +
                         f["psar_dir"].fillna(0) +
                         np.sign(f["ma_ribbon"]).fillna(0)) / 4.0

    return f.fillna(0).clip(-10, 10)


# ════════════════════════════════════════════════════════════════
#   Multi-timeframe builder
# ════════════════════════════════════════════════════════════════

def build_features(m1_df: pd.DataFrame, m5_df: pd.DataFrame, m15_df: pd.DataFrame) -> pd.DataFrame:
    """Returns 90 features: 29/TF × 3 TFs + 3 session features."""
    print("  Computing M1 features (incl. experts)...")
    f1  = compute_features(m1_df).add_prefix("m1_")
    print("  Computing M5 features (incl. experts)...")
    f5  = compute_features(m5_df).add_prefix("m5_")
    print("  Computing M15 features (incl. experts)...")
    f15 = compute_features(m15_df).add_prefix("m15_")

    f5_r  = f5.reindex(f1.index,  method="ffill")
    f15_r = f15.reindex(f1.index, method="ffill")

    hour = f1.index.hour
    dow  = f1.index.dayofweek
    session = pd.DataFrame({
        "ses_hour_sin": np.sin(2 * np.pi * hour / 24),
        "ses_hour_cos": np.cos(2 * np.pi * hour / 24),
        "ses_dow_sin":  np.sin(2 * np.pi * dow / 5),
    }, index=f1.index)

    combined = pd.concat([f1, f5_r, f15_r, session], axis=1)
    return combined.fillna(0)
