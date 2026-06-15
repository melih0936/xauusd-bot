"""
MT5 Connector — handles all MetaTrader 5 communication.
Clean, simple, no hidden magic.
"""

import os
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


class MT5Connector:
    def __init__(self):
        self.connected = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        login    = int(os.getenv("MT5_LOGIN", 0))
        password = os.getenv("MT5_PASSWORD", "")
        server   = os.getenv("MT5_SERVER", "")

        if not mt5.initialize(login=login, password=password, server=server):
            raise ConnectionError(f"MT5 init failed: {mt5.last_error()}")

        self.connected = True
        info = mt5.account_info()
        print(f"✅ MT5 connected | Account: {info.login} | "
              f"Balance: ${info.balance:,.2f} | Server: {info.server}")
        return self

    def disconnect(self):
        mt5.shutdown()
        self.connected = False
        print("🔌 MT5 disconnected")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _rates_to_df(self, rates):
        """Convert MT5 rates array to a clean DataFrame with DatetimeIndex."""
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
        df = df[["open", "high", "low", "close", "tick_volume"]].copy()
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return df

    def get_bars(self, symbol: str, timeframe: str, n_bars: int = 1000) -> pd.DataFrame:
        """Fetch the most recent N bars."""
        tf = TIMEFRAME_MAP[timeframe]
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
        if rates is None or len(rates) == 0:
            raise ValueError(f"No data for {symbol} {timeframe}: {mt5.last_error()}")
        return self._rates_to_df(rates)

    def get_bars_range(self, symbol: str, timeframe: str,
                       date_from, date_to) -> pd.DataFrame:
        """Fetch bars between two dates."""
        tf = TIMEFRAME_MAP[timeframe]
        mt5.symbol_select(symbol, True)
        # Ensure datetime objects (not date)
        if not isinstance(date_from, datetime):
            date_from = datetime(date_from.year, date_from.month, date_from.day)
        if not isinstance(date_to, datetime):
            date_to = datetime(date_to.year, date_to.month, date_to.day)
        rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
        if rates is None or len(rates) == 0:
            raise ValueError(f"No range data for {symbol} {timeframe}: {mt5.last_error()}")
        return self._rates_to_df(rates)

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        info = mt5.account_info()
        return {
            "balance":     info.balance,
            "equity":      info.equity,
            "margin_free": info.margin_free,
            "profit":      info.profit,
        }

    def get_positions(self, symbol: str):
        pos = mt5.positions_get(symbol=symbol)
        return list(pos) if pos else []

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------

    def place_order(self, symbol: str, direction: str,
                    lot: float, sl: float, tp: float,
                    comment: str = "XAUUSD-AI") -> bool:
        tick = mt5.symbol_info_tick(symbol)
        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        request = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      symbol,
            "volume":      lot,
            "type":        order_type,
            "price":       price,
            "sl":          sl,
            "tp":          tp,
            "deviation":   20,
            "magic":       20260615,
            "comment":     comment,
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == 10009:
            print(f"  ✅ {direction} {lot} lots @ {price:.2f} | SL:{sl:.2f} TP:{tp:.2f}")
            return True
        else:
            code = result.retcode if result else "None"
            msg  = result.comment if result else "No response"
            print(f"  ❌ Order failed [{code}]: {msg}")
            return False

    def close_position(self, position) -> bool:
        tick = mt5.symbol_info_tick(position.symbol)
        if position.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      position.symbol,
            "volume":      position.volume,
            "type":        order_type,
            "position":    position.ticket,
            "price":       price,
            "deviation":   20,
            "magic":       20260615,
            "comment":     "AI-Close",
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return result and result.retcode == 10009
