# -*- coding: utf-8 -*-
"""
TFT 학습 골조 (CPU)
- 기존 run_fit.py/run_tape.py의 구조를 Torch로 옮기기 쉽게 만든 템플릿
- 실제 데이터 로드(pickle/gzip 등)는 TODO로 남겨두고, 인터페이스만 확정
"""

from __future__ import annotations
import os
import math
import gzip
import pickle
import pandas as pd
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tft_model import TFTConfig, TemporalFusionTransformer, SimpleLSTM


# -----------------------------
# Dataset
# -----------------------------
class SeqDataset(Dataset):
    """
    기대 형태:
      x_past:  (N, L, past_vars)
      x_known: (N, L, known_vars)  (없으면 None)
      x_static:(N, static_vars)     (없으면 None)
      pos:     (N,)                (없으면 None)
      y:       (N, output_dim) 또는 (N,) (multiclass는 class index)
    """
    def __init__(self, x_past, y, x_known=None, x_static=None, pos=None):
        self.x_past = x_past
        self.x_known = x_known
        self.x_static = x_static
        self.pos = pos
        self.y = y

    def __len__(self):
        return self.x_past.shape[0]

    def __getitem__(self, i):
        item = {
            "x_past": torch.tensor(self.x_past[i], dtype=torch.float32),
            "y": torch.tensor(self.y[i]),
        }
        if self.x_known is not None:
            item["x_known"] = torch.tensor(self.x_known[i], dtype=torch.float32)
        else:
            item["x_known"] = None
        if self.x_static is not None:
            item["x_static"] = torch.tensor(self.x_static[i], dtype=torch.float32)
        else:
            item["x_static"] = None
        if self.pos is not None:
            item["pos"] = torch.tensor(self.pos[i], dtype=torch.long)
        else:
            item["pos"] = None
        return item


def collate_fn(batch):
    # None 처리 때문에 custom collate
    def stack_or_none(key, dtype=None):
        vals = [b[key] for b in batch]
        if vals[0] is None:
            return None
        if dtype is not None:
            return torch.stack([v.to(dtype) for v in vals], dim=0)
        return torch.stack(vals, dim=0)

    x_past = stack_or_none("x_past", torch.float32)
    x_known = stack_or_none("x_known", torch.float32)
    x_static = stack_or_none("x_static", torch.float32)
    pos = stack_or_none("pos", torch.long)
    y = stack_or_none("y")
    return x_past, x_known, x_static, pos, y


# -----------------------------
# Train / Eval
# -----------------------------
def get_loss_fn(cfg: TFTConfig):
    if cfg.output_mode == "regression" or cfg.output_mode == "binary":
        return nn.MSELoss() if cfg.output_mode == "regression" else nn.BCELoss()
    if cfg.output_mode == "multiclass":
        return nn.CrossEntropyLoss()
    raise ValueError(cfg.output_mode)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    n = 0
    for x_past, x_known, x_static, pos, y in loader:
        x_past = x_past.to(device)
        if x_known is not None: x_known = x_known.to(device)
        if x_static is not None: x_static = x_static.to(device)
        if pos is not None: pos = pos.to(device)

        y = y.to(device)
        y_hat, _ = model(x_past, x_known=x_known, x_static=x_static, pos=pos)

        # multiclass: y should be (B,) long
        if model.cfg.output_mode == "multiclass":
            if y.dim() > 1:
                y = y.squeeze(-1)
            y = y.long()
        else:
            y = y.float()
            y_hat = y_hat.float()

        loss = loss_fn(y_hat, y) if model.cfg.output_mode != "multiclass" else loss_fn(y_hat, y)
        total += float(loss.item()) * x_past.size(0)
        n += x_past.size(0)
    return total / max(n, 1)


def train(
    cfg: TFTConfig,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader : DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    save_path: str = "tft_best.pt",
):
    device = torch.device("cpu")  # CPU 고정
    model = TemporalFusionTransformer(cfg).to(device)

    loss_fn = get_loss_fn(cfg)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.2, patience=5, verbose=True)

    best = float("inf")
    bad = 0
    df_epoch = pd.DataFrame()
    for epoch in range(1, epochs + 1):
        model.train()
        for x_past, x_known, x_static, pos, y in train_loader:
            x_past = x_past.to(device)
            if x_known is not None: x_known = x_known.to(device)
            if x_static is not None: x_static = x_static.to(device)
            if pos is not None: pos = pos.to(device)
            y = y.to(device)

            y_hat, _ = model(x_past, x_known=x_known, x_static=x_static, pos=pos) #여기 attention score받을까

            if cfg.output_mode == "multiclass":
                if y.dim() > 1:
                    y = y.squeeze(-1)
                y = y.long()
                loss = loss_fn(y_hat, y)
            else:
                y = y.float()
                loss = loss_fn(y_hat.float(), y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

        tr = evaluate(model, train_loader, loss_fn, device)
        val = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(val)

        ts = evaluate(model, test_loader, loss_fn, device)

        print(f"[TFT Epoch {epoch:03d}] train_loss={tr:.6f} | val_loss={val:.6f} | test_loss={ts:.6f} | lr={optim.param_groups[0]['lr']:.2e}")

        if val < best - 1e-8:
            best = val
            bad = 0
            torch.save({"cfg": asdict(cfg), "state_dict": model.state_dict()}, save_path)
            print(f"  -> saved best to {save_path}")
        else:
            bad += 1
            if bad >= patience:
                print("  -> early stopping")
                break
        df_tmp = pd.DataFrame({'epoch':[epoch],
                               'train_loss':[tr],
                               'val_loss':[val],
                               'test_loss':[ts],
                               'lr': [optim.param_groups[0]['lr']]
                               })
        df_epoch = pd.concat([df_epoch,df_tmp])
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model, df_epoch

def train_lstm(
    input_dim: int,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader : DataLoader,
    hidden_dim: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    save_path: str = "lstm_best.pt"
):
    device = torch.device("cpu")
    # SimpleLSTM 모델 초기화
    model = SimpleLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1).to(device)
    
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.2, patience=5)

    best = float("inf")
    bad = 0
    df_epoch = pd.DataFrame()
    for epoch in range(1, epochs + 1):
        model.train()
        for x_past, x_known, x_static, pos, y in train_loader:
            x_past, y = x_past.to(device), y.to(device).float()

            y_hat, _ = model(x_past)
            loss = loss_fn(y_hat, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

        # 검증 및 테스트 평가
        tr = evaluate(model, train_loader, loss_fn, device)
        val = evaluate(model, valid_loader, loss_fn, device)
        ts = evaluate(model, test_loader, loss_fn, device)
        scheduler.step(val)
        print(f"[LSTM Epoch {epoch:03d}] train_loss={tr:.6f} | val_loss={val:.6f} | test_loss={ts:.6f} | lr={optim.param_groups[0]['lr']:.2e}")

        if val < best - 1e-8:
            best = val
            bad = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved best LSTM to {save_path}")
        else:
            bad += 1
            if bad >= patience:
                print("  -> early stopping")
                break
        df_tmp = pd.DataFrame({'epoch':[epoch],
                               'train_loss':[tr],
                               'val_loss':[val],
                               'test_loss':[ts],
                               'lr': [optim.param_groups[0]['lr']]
                               })
        df_epoch = pd.concat([df_epoch,df_tmp], ignore_index=False)

    return model, df_epoch


def main():
    print('참조\n (1) run_regression.py \n (2) prepare_data.py')
    raise SystemExit(
        "데이터 로드/피처 분해(past/known/static) 작성후 돌아감"
    )


if __name__ == "__main__":
    main()
