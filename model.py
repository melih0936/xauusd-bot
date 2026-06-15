"""
AI Model — 2-layer LSTM that predicts BUY / SELL / HOLD.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


DIRECTIONS = ["BUY", "SELL", "HOLD"]


# ------------------------------------------------------------------
# PyTorch model
# ------------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_size, 3)   # BUY | SELL | HOLD

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # Take the last timestep
        out = self.norm(out)
        out = self.dropout(out)
        return self.head(out)


# ------------------------------------------------------------------
# High-level wrapper
# ------------------------------------------------------------------

class AITrader:
    def __init__(self, input_size: int = 60, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        self.device      = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.trained     = False
        self.model       = LSTMModel(input_size, hidden_size, num_layers, dropout).to(self.device)
        print(f"🤖 AITrader | device={self.device} | features={input_size}")

    # ------------------------------------------------------------------

    def save(self, path: str = "model.pt"):
        torch.save({
            "model_state": self.model.state_dict(),
            "input_size":  self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers":  self.num_layers,
        }, path)
        print(f"💾 Saved → {path}")

    def load(self, path: str = "model.pt") -> bool:
        if not Path(path).exists():
            print(f"⚠️  No model at {path} — run train.py first")
            return False
        ck = torch.load(path, map_location=self.device)
        self.input_size  = ck["input_size"]
        self.hidden_size = ck["hidden_size"]
        self.num_layers  = ck["num_layers"]
        self.model = LSTMModel(self.input_size, self.hidden_size, self.num_layers).to(self.device)
        self.model.load_state_dict(ck["model_state"])
        self.trained = True
        print(f"✅ Loaded ← {path}")
        return True

    # ------------------------------------------------------------------

    def predict(self, seq: np.ndarray):
        """
        seq: numpy array of shape (seq_len, n_features)
        Returns: (direction: str, confidence: float, probs: np.ndarray)
        """
        self.model.eval()
        with torch.no_grad():
            x      = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx        = int(probs.argmax())
        direction  = DIRECTIONS[idx]
        confidence = float(probs[idx])
        return direction, confidence, probs
