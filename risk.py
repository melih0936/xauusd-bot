"""
Risk Manager — position sizing, SL/TP, daily limits.
"""

import MetaTrader5 as mt5


class RiskManager:
    def __init__(self, risk_pct: float = 0.01, atr_sl_mult: float = 2.0,
                 rr_ratio: float = 2.0, max_positions: int = 3,
                 daily_dd_limit: float = 0.03):
        self.risk_pct       = risk_pct
        self.atr_sl_mult    = atr_sl_mult
        self.rr_ratio       = rr_ratio
        self.max_positions  = max_positions
        self.daily_dd_limit = daily_dd_limit
        self._start_balance = None   # Reset each day

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def reset_day(self, balance: float):
        self._start_balance = balance
        print(f"📅 New day — starting balance: ${balance:,.2f}")

    # ------------------------------------------------------------------
    # Gate checks
    # ------------------------------------------------------------------

    def can_trade(self, balance: float, n_positions: int) -> tuple[bool, str]:
        """Returns (allowed, reason)."""
        if self._start_balance is None:
            self._start_balance = balance

        dd = (self._start_balance - balance) / self._start_balance
        if dd >= self.daily_dd_limit:
            return False, f"Daily drawdown {dd:.1%} ≥ limit {self.daily_dd_limit:.1%}"

        if n_positions >= self.max_positions:
            return False, f"Max positions {n_positions}/{self.max_positions} reached"

        return True, "OK"

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def lot_size(self, symbol: str, balance: float, sl_distance: float) -> float:
        """
        Risk a fixed % of balance per trade.
        sl_distance is in price units (e.g. 3.50 for Gold).
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            return info.volume_min if info else 0.01

        risk_amount = balance * self.risk_pct
        # tick_value = profit per 1 lot per tick movement
        ticks_in_sl = sl_distance / info.trade_tick_size
        lot = risk_amount / (ticks_in_sl * info.trade_tick_value)
        lot = round(lot, 2)
        lot = max(info.volume_min, min(info.volume_max, lot))
        return lot

    # ------------------------------------------------------------------
    # SL / TP
    # ------------------------------------------------------------------

    def sl_tp(self, symbol: str, direction: str, price: float, atr_val: float):
        """
        Returns (sl_price, tp_price, sl_distance).
        sl = price ± ATR * multiplier
        tp = sl_distance * RR ratio
        """
        sl_dist = atr_val * self.atr_sl_mult
        tp_dist = sl_dist * self.rr_ratio

        if direction == "BUY":
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist

        digits = mt5.symbol_info(symbol).digits
        return round(sl, digits), round(tp, digits), sl_dist
