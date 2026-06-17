"""
XAUUSD AI Trading Bot — Main Loop
Runs every 60 seconds, evaluates M1/M5/M15 signals, trades on demo.

Usage:
    py -3.11 bot.py
"""

import pickle
import time
from datetime import date, datetime

import numpy as np

import config
from connector import MT5Connector
from features import build_features
from model import AITrader
from news_filter import is_news_blackout
from risk import RiskManager


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def current_atr(m5_df, period: int = 14) -> float:
    """ATR from the M5 chart — used for SL/TP calculation."""
    from features import atr as _atr
    return float(_atr(m5_df["high"], m5_df["low"], m5_df["close"], period).iloc[-1])


def bar(label: str, value, width: int = 40) -> str:
    filled = int(value * width)
    return f"{label} [{'█' * filled}{'░' * (width - filled)}] {value:.1%}"


# ──────────────────────────────────────────────────────────────────
# Single evaluation cycle
# ──────────────────────────────────────────────────────────────────

def evaluate(connector: MT5Connector, trader: AITrader,
             scaler, risk: RiskManager):

    # 1. Fetch live multi-TF data
    m1_df  = connector.get_bars(config.SYMBOL, "M1",  n_bars=500)
    m5_df  = connector.get_bars(config.SYMBOL, "M5",  n_bars=300)
    m15_df = connector.get_bars(config.SYMBOL, "M15", n_bars=200)

    # 2. Features → normalise → sequence
    feat_df    = build_features(m1_df, m5_df, m15_df)
    feat_norm  = scaler.transform(feat_df.values)
    seq        = feat_norm[-config.SEQ_LEN:]

    if len(seq) < config.SEQ_LEN:
        print("⚠️  Not enough bars yet — skipping.")
        return

    # 3. Predict
    direction, confidence, probs = trader.predict(seq)
    buy_p, sell_p, hold_p = probs

    # 4. Account snapshot
    acct      = connector.get_account()
    balance   = acct["balance"]
    positions = connector.get_positions(config.SYMBOL)
    atr_val   = current_atr(m5_df)
    price     = float(m5_df["close"].iloc[-1])

    # 5. Log
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'─'*58}")
    print(f" {ts}  {config.SYMBOL}  ${price:,.2f}  |  Balance: ${balance:,.2f}")
    print(f" {bar('BUY ', buy_p)}")
    print(f" {bar('SELL', sell_p)}")
    print(f" {bar('HOLD', hold_p)}")
    print(f" Signal: {direction}  ({confidence:.1%})  |  Open positions: {len(positions)}")

    # 6. Trade decision
    if direction == "HOLD":
        print(" ⏸️  HOLD — no trade")
        return

    if confidence < config.MIN_CONFIDENCE:
        print(f" ⏸️  Confidence {confidence:.1%} < {config.MIN_CONFIDENCE:.1%} — skipping")
        return

    allowed, reason = risk.can_trade(balance, len(positions))
    if not allowed:
        print(f" 🛑  Blocked: {reason}")
        return

    blackout, event_name, mins_away = is_news_blackout(m1_df.index[-1])
    if blackout:
        print(f" 📰  News blackout: {event_name} ~{abs(mins_away):.0f}min away — skipping")
        return

    # 7. Place order
    sl, tp, sl_dist = risk.sl_tp(config.SYMBOL, direction, price, atr_val)
    lot             = risk.lot_size(config.SYMBOL, balance, sl_dist, confidence)

    print(f" 🔔  {direction} | lot={lot} | SL={sl:.2f} | TP={tp:.2f} | ATR={atr_val:.2f}")
    connector.place_order(config.SYMBOL, direction, lot, sl, tp)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("   XAUUSD AI Trading Bot")
    print(f"   Symbol: {config.SYMBOL}  |  TFs: M1 / M5 / M15")
    print(f"   Risk: {config.RISK_PER_TRADE:.0%}/trade  |  "
          f"Min confidence: {config.MIN_CONFIDENCE:.0%}")
    print("=" * 58)

    # Load everything
    connector = MT5Connector().connect()

    trader = AITrader()
    if not trader.load(config.MODEL_PATH):
        print("\n❌ No trained model found. Run  py -3.11 train.py  first.")
        connector.disconnect()
        return

    with open(config.SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    risk = RiskManager(
        risk_pct       = config.RISK_PER_TRADE,
        atr_sl_mult    = config.ATR_SL_MULTIPLIER,
        rr_ratio       = config.RR_RATIO,
        max_positions  = config.MAX_POSITIONS,
        daily_dd_limit = config.DAILY_DRAWDOWN_LIMIT,
    )

    print("\n✅ Bot is live — press Ctrl+C to stop\n")

    today = date.today()

    try:
        while True:
            # Daily reset
            if date.today() != today:
                today = date.today()
                acct  = connector.get_account()
                risk.reset_day(acct["balance"])

            try:
                evaluate(connector, trader, scaler, risk)
            except Exception as e:
                print(f"⚠️  Cycle error: {e}")

            # Sync precisely to the NEXT M1 bar close — not a flat 60s sleep.
            # A flat sleep drifts later every cycle by however long evaluate()
            # took to run. This instead always wakes up ~1s after each new
            # minute boundary, so we never lag behind the actual bar close.
            now = datetime.now()
            seconds_into_minute = now.second + now.microsecond / 1_000_000
            wait = max(1.0, 61.0 - seconds_into_minute)
            while wait > 0:
                print(f"\r ⏳ Next bar close in {wait:4.1f}s ...", end="", flush=True)
                step = min(1.0, wait)
                time.sleep(step)
                wait -= step

    except KeyboardInterrupt:
        print("\n\n🛑 Bot stopped by user.")
    finally:
        connector.disconnect()


if __name__ == "__main__":
    main()
