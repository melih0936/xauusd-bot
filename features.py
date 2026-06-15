"""
Feature Engineering — Advanced Multi-Timeframe Indicators
~45 features per timeframe × 3 timeframes = ~135 total features

Indicators:
  Core        — Returns, RSI, MACD, Bollinger Bands, ATR, Stochastic
  Trend       — EMA (9/21/50/100/200), ADX, EMA crossovers
  Oscillators — CCI, MFI, ROC, Williams %R
  Volume      — CVD, Volume Profile (POC/VAH/VAL), Volume Ratio, MFI
  VWAP        — Daily VWAP + bands (±1σ, ±2σ)
  S/R Levels  — Swing highs/lows, Donchian channels, breakout signals
  Candle      — Body, wicks, engulfing patterns
  Session     — Hour of day, day of week (cyclically encoded)
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────
# Base indicator functions
# ──────────────────────────────────────────────────────────────────

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

def cci(high, low, close, period=20):
    tp  = (high + low + close) / 3
    ma  = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return ((tp - ma) / (0.015 * mad + 1e-10)) / 200

def mfi(high, low, close, volume, period=14):
    tp  = (high + low + close) / 3
    mf  = tp * volume
    pos = mf.where(tp > tp.shift(), 0).rolling(period).sum()
    neg = mf.where(tp < tp.shift(), 0).rolling(period).sum()
    return pos / (pos + neg + 1e-10)


# ──────────────────────────────────────────────────────────────────
# Advanced indicators
# ──────────────────────────────────────────────────────────────────

def vwap_daily(high, low, close, volume):
    """
    Daily VWAP — resets at midnight each day.
    Returns VWAP, upper band (+1σ, +2σ), lower band (-1σ, -2σ).
    """
    typical = (high + low + close) / 3
    df_tmp  = pd.DataFrame({
        "tp": typical, "vol": volume,
        "tpvol": typical * volume, "tp2vol": (typical ** 2) * volume
    })

    results = {"vwap": [], "vwap_u1": [], "vwap_l1": [],
               "vwap_u2": [], "vwap_l2": []}

    for date, grp in df_tmp.groupby(df_tmp.index.date):
        cum_tpvol = grp["tpvol"].cumsum()
        cum_tp2vol = grp["tp2vol"].cumsum()
        cum_vol   = grp["vol"].cumsum()
        vw = cum_tpvol / (cum_vol + 1e-10)
        variance = cum_tp2vol / (cum_vol + 1e-10) - vw ** 2
        std  = np.sqrt(variance.clip(lower=0))
        for idx in grp.index:
            i = grp.index.get_loc(idx)
            results["vwap"].append((idx, vw.iloc[i]))
            results["vwap_u1"].append((idx, vw.iloc[i] + std.iloc[i]))
            results["vwap_l1"].append((idx, vw.iloc[i] - std.iloc[i]))
            results["vwap_u2"].append((idx, vw.iloc[i] + 2*std.iloc[i]))
            results["vwap_l2"].append((idx, vw.iloc[i] - 2*std.iloc[i]))

    def to_series(pairs):
        idx, vals = zip(*pairs)
        return pd.Series(vals, index=idx)

    return (to_series(results["vwap"]),
            to_series(results["vwap_u1"]),
            to_series(results["vwap_l1"]),
            to_series(results["vwap_u2"]),
            to_series(results["vwap_l2"]))


def cvd(close, open_, volume, period=20):
    """
    Cumulative Volume Delta — approximates buying vs selling pressure.
    Delta per bar = volume × sign(close - open)
    CVD = rolling cumulative sum of delta.
    """
    delta   = volume * np.sign(close - open_)
    cvd_raw = delta.rolling(period).sum()
    # Normalize by average volume
    avg_vol = volume.rolling(period).mean() * period
    cvd_norm = cvd_raw / (avg_vol + 1e-10)
    cvd_slope = cvd_norm.diff(5)          # 5-bar slope of CVD
    return cvd_norm, cvd_slope


def volume_profile(high, low, close, volume, period=100, n_bins=20):
    """
    Simplified Volume Profile over a rolling window.
    Returns distance to: POC (point of control), VAH, VAL.
    POC  = price level with most volume
    VAH  = top of 70% value area
    VAL  = bottom of 70% value area
    """
    poc_dist  = pd.Series(index=close.index, dtype=float)
    vah_dist  = pd.Series(index=close.index, dtype=float)
    val_dist  = pd.Series(index=close.index, dtype=float)

    for i in range(period, len(close)):
        window_h = high.iloc[i-period:i]
        window_l = low.iloc[i-period:i]
        window_c = close.iloc[i-period:i]
        window_v = volume.iloc[i-period:i]

        price_min = window_l.min()
        price_max = window_h.max()
        if price_max == price_min:
            continue

        bins = np.linspace(price_min, price_max, n_bins + 1)
        bin_vol = np.zeros(n_bins)
        typical = (window_h + window_l + window_c) / 3

        for j in range(len(typical)):
            b = int((typical.iloc[j] - price_min) /
                    (price_max - price_min) * n_bins)
            b = min(b, n_bins - 1)
            bin_vol[b] += window_v.iloc[j]

        poc_bin  = int(np.argmax(bin_vol))
        poc_price = (bins[poc_bin] + bins[poc_bin + 1]) / 2

        # Value Area (70% of total volume)
        total_vol  = bin_vol.sum()
        target_vol = total_vol * 0.70
        va_bins    = [poc_bin]
        accum      = bin_vol[poc_bin]
        lo, hi     = poc_bin, poc_bin

        while accum < target_vol:
            up_vol   = bin_vol[hi + 1] if hi + 1 < n_bins else 0
            down_vol = bin_vol[lo - 1] if lo - 1 >= 0 else 0
            if up_vol >= down_vol and hi + 1 < n_bins:
                hi += 1; accum += bin_vol[hi]
            elif lo - 1 >= 0:
                lo -= 1; accum += bin_vol[lo]
            else:
                break

        vah_price = (bins[hi] + bins[hi + 1]) / 2
        val_price = (bins[lo] + bins[lo + 1]) / 2

        cur = close.iloc[i]
        atr_val = atr(high, low, close, 14).iloc[i]
        poc_dist.iloc[i]  = (cur - poc_price)  / (atr_val + 1e-10)
        vah_dist.iloc[i]  = (cur - vah_price)  / (atr_val + 1e-10)
        val_dist.iloc[i]  = (cur - val_price)  / (atr_val + 1e-10)

    return poc_dist.fillna(0), vah_dist.fillna(0), val_dist.fillna(0)


def support_resistance(high, low, close, period=20, lookback=200):
    """
    Swing high/low based S/R.
    Returns distance (in ATR units) to nearest support and resistance.
    """
    # Swing highs: local maxima
    swing_high = high.rolling(period, center=True).max() == high
    swing_low  = low.rolling(period,  center=True).min() == low

    sr_dist   = pd.Series(index=close.index, dtype=float)
    sup_dist  = pd.Series(index=close.index, dtype=float)

    atr14 = atr(high, low, close, 14)

    for i in range(lookback, len(close)):
        cur = close.iloc[i]
        av  = atr14.iloc[i]

        # Resistance: nearest swing high above price
        highs_above = high[swing_high].iloc[max(0, i-lookback):i]
        highs_above = highs_above[highs_above > cur]
        res = (highs_above.min() - cur) / (av + 1e-10) if len(highs_above) > 0 else 5.0

        # Support: nearest swing low below price
        lows_below = low[swing_low].iloc[max(0, i-lookback):i]
        lows_below = lows_below[lows_below < cur]
        sup = (cur - lows_below.max()) / (av + 1e-10) if len(lows_below) > 0 else 5.0

        sr_dist.iloc[i]  = min(res, 5.0)   # Distance to nearest resistance
        sup_dist.iloc[i] = min(sup, 5.0)   # Distance to nearest support

    return sr_dist.fillna(5.0), sup_dist.fillna(5.0)


def donchian_breakout(high, low, close, period=20):
    """
    Donchian channel — detects breakouts.
    Returns: channel position (0-1) and breakout signal (-1/0/+1).
    """
    don_high = high.rolling(period).max()
    don_low  = low.rolling(period).min()
    rng      = don_high - don_low + 1e-10

    # Position within channel (0=at bottom, 1=at top)
    position   = (close - don_low) / rng

    # Breakout: +1 if new high, -1 if new low, 0 otherwise
    breakout = pd.Series(0.0, index=close.index)
    breakout[close >= don_high.shift(1)] =  1.0
    breakout[close <= don_low.shift(1)]  = -1.0

    return position, breakout


# ──────────────────────────────────────────────────────────────────
# Full per-timeframe feature computation (~45 features)
# ──────────────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame, use_volume_profile: bool = False) -> pd.DataFrame:
    """
    Computes all features for a single timeframe.
    set use_volume_profile=False for faster training (VP is slow).
    """
    f = pd.DataFrame(index=df.index)
    c = df["close"]; h = df["high"]; l = df["low"]
    o = df["open"];  v = df["volume"]

    # ── Returns ───────────────────────────────────────── 3
    f["ret1"]  = c.pct_change(1)
    f["ret5"]  = c.pct_change(5)
    f["ret10"] = c.pct_change(10)

    # ── EMA distances (5 EMAs) ───────────────────────── 5
    for span in [9, 21, 50, 100, 200]:
        f[f"ema{span}"] = c.ewm(span=span).mean() / c - 1

    # ── EMA crossover signals ────────────────────────── 2
    f["ema9x21"]   = (c.ewm(span=9).mean()  - c.ewm(span=21).mean()) / c
    f["ema21x50"]  = (c.ewm(span=21).mean() - c.ewm(span=50).mean()) / c

    # ── RSI ───────────────────────────────────────────── 1
    f["rsi14"] = rsi(c, 14)

    # ── MACD ─────────────────────────────────────────── 3
    m = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    s = m.ewm(span=9).mean()
    f["macd"] = m / c;  f["macd_sig"] = s / c;  f["macd_hist"] = (m - s) / c

    # ── Bollinger Bands ───────────────────────────────── 3
    bm  = c.rolling(20).mean();  bsd = c.rolling(20).std()
    f["bb_up"]  = (bm + 2*bsd - c) / c
    f["bb_lo"]  = (c - (bm - 2*bsd)) / c
    f["bb_pct"] = (c - (bm - 2*bsd)) / (4*bsd + 1e-10)

    # ── ATR ──────────────────────────────────────────── 1
    atr14 = atr(h, l, c, 14)
    f["atr14"] = atr14 / c

    # ── Stochastic + Williams %R ──────────────────────── 3
    lo14 = l.rolling(14).min();  hi14 = h.rolling(14).max()
    stk  = (c - lo14) / (hi14 - lo14 + 1e-10)
    f["stoch_k"] = stk;  f["stoch_d"] = stk.rolling(3).mean()
    f["wpr"]     = (hi14 - c) / (hi14 - lo14 + 1e-10)

    # ── ADX ──────────────────────────────────────────── 1
    f["adx14"] = adx(h, l, c, 14)

    # ── CCI ──────────────────────────────────────────── 1
    f["cci20"] = cci(h, l, c, 20).clip(-2, 2) / 2

    # ── MFI ──────────────────────────────────────────── 1
    f["mfi14"] = mfi(h, l, c, v, 14)

    # ── ROC ──────────────────────────────────────────── 1
    f["roc10"] = c.pct_change(10)

    # ── CVD ──────────────────────────────────────────── 2
    cvd_val, cvd_slp = cvd(c, o, v, period=20)
    f["cvd"]       = cvd_val.clip(-3, 3)
    f["cvd_slope"] = cvd_slp.clip(-1, 1)

    # ── VWAP ─────────────────────────────────────────── 5
    try:
        vw, vu1, vl1, vu2, vl2 = vwap_daily(h, l, c, v)
        f["vwap_dist"] = (c - vw)  / (atr14 + 1e-10)
        f["vwap_u1"]   = (c - vu1) / (atr14 + 1e-10)
        f["vwap_l1"]   = (c - vl1) / (atr14 + 1e-10)
        f["vwap_u2"]   = (c - vu2) / (atr14 + 1e-10)
        f["vwap_l2"]   = (c - vl2) / (atr14 + 1e-10)
    except Exception:
        for k in ["vwap_dist","vwap_u1","vwap_l1","vwap_u2","vwap_l2"]:
            f[k] = 0.0

    # ── Donchian / Breakout ───────────────────────────── 2
    don_pos, don_brk = donchian_breakout(h, l, c, 20)
    f["don_pos"] = don_pos
    f["don_brk"] = don_brk

    # ── S/R distances ────────────────────────────────── 2
    try:
        res_dist, sup_dist = support_resistance(h, l, c, period=10, lookback=100)
        f["res_dist"] = res_dist.clip(0, 5) / 5
        f["sup_dist"] = sup_dist.clip(0, 5) / 5
    except Exception:
        f["res_dist"] = 0.0;  f["sup_dist"] = 0.0

    # ── Volume Profile ───────────────────────────────── 3 (optional — slow)
    if use_volume_profile:
        try:
            poc, vah, val_ = volume_profile(h, l, c, v, period=50, n_bins=15)
            f["vp_poc"] = poc.clip(-5, 5);  f["vp_vah"] = vah.clip(-5, 5)
            f["vp_val"] = val_.clip(-5, 5)
        except Exception:
            f["vp_poc"] = 0.0;  f["vp_vah"] = 0.0;  f["vp_val"] = 0.0

    # ── Volume ratio ─────────────────────────────────── 1
    f["vol_ratio"] = v / (v.rolling(20).mean() + 1e-10)

    # ── Candle structure ─────────────────────────────── 3
    rng  = (h - l + 1e-10)
    f["body"]       = (c - o) / rng
    f["upper_wick"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / rng
    f["lower_wick"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / rng

    # ── Session / Time ────────────────────────────────── 3
    hour = df.index.hour
    dow  = df.index.dayofweek
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["dow_sin"]  = np.sin(2 * np.pi * dow / 5)

    return f.fillna(0).clip(-10, 10)


# ──────────────────────────────────────────────────────────────────
# Multi-timeframe builder
# ──────────────────────────────────────────────────────────────────

def build_features(m1_df: pd.DataFrame,
                   m5_df: pd.DataFrame,
                   m15_df: pd.DataFrame,
                   use_volume_profile: bool = False) -> pd.DataFrame:
    """
    Returns a DataFrame with ~135 features (45 per TF × 3 TFs).
    Set use_volume_profile=True for full accuracy (slower).
    """
    print("  Computing M1 features...")
    f1  = compute_features(m1_df,  use_volume_profile).add_prefix("m1_")
    print("  Computing M5 features...")
    f5  = compute_features(m5_df,  use_volume_profile).add_prefix("m5_")
    print("  Computing M15 features...")
    f15 = compute_features(m15_df, use_volume_profile).add_prefix("m15_")

    f5_r  = f5.reindex(f1.index,  method="ffill")
    f15_r = f15.reindex(f1.index, method="ffill")

    combined = pd.concat([f1, f5_r, f15_r], axis=1)
    return combined.fillna(0)
