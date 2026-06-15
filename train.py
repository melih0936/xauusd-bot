"""
Training Script — fetches data, builds features, trains LSTM, saves model.
Run this once before starting the bot.

Usage:
    py -3.11 train.py
"""

import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import config
from connector import MT5Connector
from features import build_features
from model import AITrader, LSTMModel


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------------------------------------------------------
# Label generation
# ------------------------------------------------------------------

def make_labels(close, forward_bars: int = 10, threshold: float = 0.003) -> np.ndarray:
    """
    BUY  (0): price rises > threshold% in next N bars
    SELL (1): price falls > threshold% in next N bars
    HOLD (2): price stays flat
    """
    future_ret = close.shift(-forward_bars) / close - 1
    labels = np.where(future_ret >  threshold, 0,
             np.where(future_ret < -threshold, 1, 2))
    return labels.astype(int)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def train():
    print("=" * 60)
    print("   XAUUSD AI Bot — Training")
    print("=" * 60)

    # 1. Fetch data
    print("\n📊 Fetching historical data from MT5...")
    conn = MT5Connector().connect()
    n1   = config.TRAIN_BARS
    n5   = max(n1 // 5,  500)
    n15  = max(n1 // 15, 200)
    m1_df  = conn.get_bars(config.SYMBOL, "M1",  n_bars=n1)
    m5_df  = conn.get_bars(config.SYMBOL, "M5",  n_bars=n5)
    m15_df = conn.get_bars(config.SYMBOL, "M15", n_bars=n15)
    conn.disconnect()
    print(f"   M1:{len(m1_df):,}  M5:{len(m5_df):,}  M15:{len(m15_df):,} bars")

    # 2. Build features
    print("\n⚙️  Building multi-timeframe features...")
    feat_df = build_features(m1_df, m5_df, m15_df)
    print(f"   Feature shape: {feat_df.shape}  ({feat_df.shape[1]} features)")

    # 3. Labels (aligned to M1)
    m5_close_aligned = m5_df["close"].reindex(feat_df.index, method="ffill")
    raw_labels = make_labels(m5_close_aligned,
                             forward_bars=config.LABEL_BARS,
                             threshold=config.LABEL_THRESHOLD)

    # Trim last rows (future unknown)
    trim = config.LABEL_BARS + 5
    feat_vals = feat_df.values[:-trim]
    labels    = raw_labels[:-trim]

    # 4. Normalise
    scaler     = StandardScaler()
    feat_norm  = scaler.fit_transform(feat_vals)
    with open(config.SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"   Scaler saved → {config.SCALER_PATH}")

    # 5. Build sequences
    print("\n🔗 Building sequences...")
    seq_len = config.SEQ_LEN
    X, y = [], []
    for i in range(seq_len, len(feat_norm)):
        X.append(feat_norm[i - seq_len : i])
        y.append(labels[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"   Sequences: {len(X):,}  Shape: {X.shape}")

    # Class distribution
    for idx, name in enumerate(["BUY", "SELL", "HOLD"]):
        pct = (y == idx).mean() * 100
        print(f"   {name}: {pct:.1f}%")

    # 6. Train / val split (80/20, no shuffle across time)
    split     = int(len(X) * 0.8)
    X_tr, X_v = X[:split], X[split:]
    y_tr, y_v = y[:split], y[split:]

    tr_loader = DataLoader(SequenceDataset(X_tr, y_tr),
                           batch_size=config.BATCH_SIZE, shuffle=True)
    v_loader  = DataLoader(SequenceDataset(X_v,  y_v),
                           batch_size=config.BATCH_SIZE)

    # 7. Model
    input_size = X.shape[2]
    trader     = AITrader(input_size=input_size,
                          hidden_size=config.HIDDEN_SIZE,
                          num_layers=config.NUM_LAYERS,
                          dropout=config.DROPOUT)

    optimizer  = torch.optim.Adam(trader.model.parameters(), lr=config.LEARNING_RATE)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                     optimizer, patience=5, factor=0.5, verbose=False)
    criterion  = nn.CrossEntropyLoss()

    # 8. Training loop
    print(f"\n🏋️  Training for {config.EPOCHS} epochs...\n")
    best_acc = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        # --- Train ---
        trader.model.train()
        total_loss = 0.0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(trader.device), yb.to(trader.device)
            optimizer.zero_grad()
            loss = criterion(trader.model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(trader.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # --- Validate ---
        trader.model.eval()
        correct = total = 0
        with torch.no_grad():
            for Xb, yb in v_loader:
                Xb, yb = Xb.to(trader.device), yb.to(trader.device)
                preds   = trader.model(Xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += len(yb)

        val_acc = correct / total
        scheduler.step(1 - val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            trader.save(config.MODEL_PATH)

        if epoch % 5 == 0 or epoch == 1:
            avg_loss = total_loss / len(tr_loader)
            print(f"  Epoch {epoch:3d}/{config.EPOCHS} | "
                  f"loss={avg_loss:.4f} | val_acc={val_acc:.3f} | "
                  f"best={best_acc:.3f} {'⭐' if val_acc == best_acc else ''}")

    print(f"\n✅ Training done!  Best val accuracy: {best_acc:.3f}")
    print(f"   Model saved → {config.MODEL_PATH}")


if __name__ == "__main__":
    train()
