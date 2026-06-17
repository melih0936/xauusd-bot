# ============================================================
#   XAUUSD AI Trading Bot — Configuration
# ============================================================

SYMBOL      = "XAUUSD"
TIMEFRAMES  = ["M1", "M5", "M15"]

# --- Model (SHRUNK to reduce overfitting) ---
SEQ_LEN      = 60
HIDDEN_SIZE  = 64       # was 256 — smaller model generalizes better
NUM_LAYERS   = 1        # was 2
DROPOUT      = 0.5      # was 0.3 — stronger regularization

# --- Training ---
TRAIN_BARS    = 150000  # was 50000 — more data fights overfitting
EPOCHS        = 40
BATCH_SIZE    = 128
LEARNING_RATE = 0.0005
WEIGHT_DECAY  = 1e-4    # NEW — L2 regularization
EARLY_STOP_PATIENCE = 8 # NEW — stop if no improvement for 8 epochs

# --- Labels ---
LABEL_THRESHOLD = 0.0015  # 0.15% move
LABEL_BARS      = 60      # look 60 M1 bars (1 hour) ahead — was 30 (too noisy)

# --- Risk Management ---
RISK_PER_TRADE       = 0.01
ATR_SL_MULTIPLIER    = 2.0
RR_RATIO             = 2.0
MAX_POSITIONS        = 3
DAILY_DRAWDOWN_LIMIT = 0.03
MIN_LOT = 0.01
MAX_LOT = 0.04

# --- Signal ---
MIN_CONFIDENCE = 0.60

# --- Files ---
MODEL_PATH  = "model.pt"
SCALER_PATH = "scaler.pkl"

# --- News Filter ---
NEWS_BLACKOUT_MINUTES = 30   # Skip trading ±30 min around high-impact events

# ⚠️ MUST VERIFY — MT5 timestamps are in your BROKER'S server time, not US Eastern.
# How to find the right value:
#   1. Run: py -3.11 -c "import MetaTrader5 as mt5; mt5.initialize(); print(mt5.symbol_info_tick('XAUUSD').time)"
#      Paste that number into unixtimestamp.com to see what time MT5 thinks it is.
#   2. Check the real current time in New York: timeanddate.com/worldclock/usa/new-york
#   3. SERVER_OFFSET_FROM_ET_HOURS = (MT5's hour) - (New York's hour)
#      Example: MT5 shows 21:15, New York is actually 14:15 → offset = 7
#   4. Re-check this in March and November when US/broker DST changes.
SERVER_OFFSET_FROM_ET_HOURS = 4   # PLACEHOLDER — replace with your real value
