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
TRAIN_BARS    = 100000  # was 50000 — more data fights overfitting
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
