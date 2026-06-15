# ============================================================
#   XAUUSD AI Trading Bot — Configuration
# ============================================================

SYMBOL      = "XAUUSD"
TIMEFRAMES  = ["M1", "M5", "M15"]

# --- Model ---
SEQ_LEN      = 60       # Bars to look back
HIDDEN_SIZE  = 256      # Increased from 128 for more capacity
NUM_LAYERS   = 2
DROPOUT      = 0.3      # Slightly higher dropout to prevent overfitting

# --- Training ---
TRAIN_BARS    = 50000
EPOCHS        = 60
BATCH_SIZE    = 128     # Larger batch = more stable gradients
LEARNING_RATE = 0.0005  # Slightly lower for better convergence

# --- Labels (FIXED — proper BUY/SELL/HOLD distribution) ---
LABEL_THRESHOLD = 0.001   # 0.1% move threshold (was 0.3% — too high for M1)
LABEL_BARS      = 30      # Look 30 M1 bars ahead = 30 minutes (was 10)

# --- Risk Management ---
RISK_PER_TRADE       = 0.01
ATR_SL_MULTIPLIER    = 2.0
RR_RATIO             = 2.0
MAX_POSITIONS        = 3
DAILY_DRAWDOWN_LIMIT = 0.03

# --- Signal ---
MIN_CONFIDENCE = 0.60

# --- Files ---
MODEL_PATH  = "model.pt"
SCALER_PATH = "scaler.pkl"
