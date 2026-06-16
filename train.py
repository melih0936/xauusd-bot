"""
Training Script — fetches data, builds features, trains LSTM, saves model.
Now with: class weighting, train/val leakage gap, weight decay, early stopping.

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
from model import AITrader


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def make_labels(close, forward_bars=60, threshold=0.0015):
    """BUY(0)/SELL(1)/HOLD(2) based on future return over `forward_bars`."""
    future_ret = close.shift(-forward_bars) / close - 1
    labels = np.where(future_ret >  threshold, 0,
             np.where(future_ret < -threshold, 1, 2))
    return labels.astype(int)


def train():
    print("=" * 60)
    print("   XAUUSD AI Bot — Training (v2: anti-overfitting)")
    print("=" * 60)

    # 1. Fetch data
    print("\n📊 Fetching historical data from MT5...")
    conn = MT5Connector().connect()
    n1  = config.TRAIN_BARS
    n5  = max(n1 // 5,  500)
    n15 = max(n1 // 15, 200)
    m1_df  = conn.get_bars(config.SYMBOL, "M1",  n_bars=n1)
    m5_df  = conn.get_bars(config.SYMBOL, "M5",  n_bars=n5)
    m15_df = conn.get_bars(config.SYMBOL, "M15", n_bars=n15)
    conn.disconnect()
    print(f"   M1:{len(m1_df):,}  M5:{len(m5_df):,}  M15:{len(m15_df):,} bars")

    # 2. Features
    print("\n⚙️  Building multi-timeframe features...")
    feat_df = build_features(m1_df, m5_df, m15_df)
    print(f"   Feature shape: {feat_df.shape}  ({feat_df.shape[1]} features)")

    # 3. Labels
    m5_close_aligned = m5_df["close"].reindex(feat_df.index, method="ffill")
    raw_labels = make_labels(m5_close_aligned,
                             forward_bars=config.LABEL_BARS,
                             threshold=config.LABEL_THRESHOLD)

    trim = config.LABEL_BARS + 5
    feat_vals = feat_df.values[:-trim]
    labels    = raw_labels[:-trim]

    print("\n   Label distribution:")
    for idx, name in enumerate(["BUY", "SELL", "HOLD"]):
        pct = (labels == idx).mean() * 100
        print(f"   {name}: {pct:.1f}%")

    # 4. Normalise
    scaler    = StandardScaler()
    feat_norm = scaler.fit_transform(feat_vals)
    with open(config.SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"   Scaler saved → {config.SCALER_PATH}")

    # 5. Sequences
    print("\n🔗 Building sequences...")
    seq_len = config.SEQ_LEN
    X, y = [], []
    for i in range(seq_len, len(feat_norm)):
        X.append(feat_norm[i - seq_len : i])
        y.append(labels[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"   Sequences: {len(X):,}  Shape: {X.shape}")

    # 6. Train/val split WITH GAP to prevent overlap leakage
    split = int(len(X) * 0.8)
    gap   = seq_len   # sequences within `gap` of the boundary overlap the other side
    X_tr, y_tr = X[:split], y[:split]
    X_v,  y_v  = X[split + gap:], y[split + gap:]
    print(f"   Train: {len(X_tr):,}  |  Val: {len(X_v):,}  (gap={gap} to prevent leakage)")

    # 7. Class weights (handle imbalance properly, instead of ignoring it)
    counts  = np.bincount(y_tr, minlength=3).astype(np.float32)
    weights = (counts.sum() / (3 * counts + 1e-6))
    weights_t = torch.FloatTensor(weights)
    print(f"   Class weights: BUY={weights[0]:.2f} SELL={weights[1]:.2f} HOLD={weights[2]:.2f}")

    tr_loader = DataLoader(SequenceDataset(X_tr, y_tr), batch_size=config.BATCH_SIZE, shuffle=True)
    v_loader  = DataLoader(SequenceDataset(X_v,  y_v),  batch_size=config.BATCH_SIZE)

    # 8. Model
    input_size = X.shape[2]
    trader = AITrader(input_size=input_size, hidden_size=config.HIDDEN_SIZE,
                      num_layers=config.NUM_LAYERS, dropout=config.DROPOUT)

    optimizer = torch.optim.Adam(trader.model.parameters(),
                                 lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=weights_t.to(trader.device))

    # 9. Training loop with early stopping
    print(f"\n🏋️  Training for up to {config.EPOCHS} epochs "
          f"(early stop patience={config.EARLY_STOP_PATIENCE})...\n")
    best_acc = 0.0
    no_improve = 0

    for epoch in range(1, config.EPOCHS + 1):
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

        trader.model.eval()
        correct = total = 0
        with torch.no_grad():
            for Xb, yb in v_loader:
                Xb, yb = Xb.to(trader.device), yb.to(trader.device)
                preds = trader.model(Xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)

        val_acc = correct / total
        scheduler.step(1 - val_acc)

        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            no_improve = 0
            trader.save(config.MODEL_PATH)
        else:
            no_improve += 1

        avg_loss = total_loss / len(tr_loader)
        print(f"  Epoch {epoch:3d}/{config.EPOCHS} | loss={avg_loss:.4f} | "
              f"val_acc={val_acc:.3f} | best={best_acc:.3f} {'⭐' if improved else ''}")

        if no_improve >= config.EARLY_STOP_PATIENCE:
            print(f"\n  ⏹️  Early stopping — no improvement for {config.EARLY_STOP_PATIENCE} epochs")
            break

    print(f"\n✅ Training done!  Best val accuracy: {best_acc:.3f}")
    print(f"   (Random guessing with 3 classes = 0.333 — anything meaningfully above that is real signal)")
    print(f"   Model saved → {config.MODEL_PATH}")


if __name__ == "__main__":
    train()
