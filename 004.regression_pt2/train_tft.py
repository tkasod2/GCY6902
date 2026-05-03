# -*- coding: utf-8 -*-
# train_tft_quantile_260427.py
"""
TFT 학습 구조 (CPU)
- 기존 run_fit.py/run_tft.py의 구조를 Torch로 옮기기 쉽게 만든 템플릿
- 실제 데이터 로드(pickle/gzip 등)는 TODO로 남겨두고, 인터페이스만 확정
"""

from __future__ import annotations
import os
import math
import gzip
import pandas as pd
# import xlsxwriter
from dataclasses import asdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tft_model import TFTConfig, TemporalFusionTransformer, SimpleLSTM
import torch
import torch.nn as nn

class MultiQuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        
    def pinball_loss(self, pred, target, q):
        errors = target - pred
        return torch.max((q - 1) * errors, q * errors)
        
    def forward(self, preds, target):
        """
        preds: [batch_size, horizon, 3] (0.1, 0.5, 0.9 순서)
        target: [batch_size, horizon]
        """
        total_loss = 0
        for i, q in enumerate(self.quantiles):
            # i번째 분위수 예측값 선택
            errors = target - preds[:, i]
            loss = torch.max((q - 1) * errors, q * errors)
            total_loss += torch.mean(loss)
            
        return total_loss / len(self.quantiles)

# -------------------------
# Dataset
# -------------------------
class SeqDataset(Dataset):
    """
    기대 형태:
    x_past: (N, L, past_vars)
    x_known: (N, L, known_vars) (없으면 None)
    x_static: (N, static_vars) (없으면 None)
    pos: (N,) (없으면 None)
    y: (N, output_dim) 또는 (N,) (multiclass는 class index)
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
        # item = {
        #     "x_past": torch.tensor(self.x_past[i], dtype=torch.float32),
        #     "y": torch.tensor(self.y[i]),
        # }
        item = {
            "x_past": torch.tensor(self.x_past[i], dtype=torch.float32),
            # y 배열이 (N, 2)이므로 0번 인덱스는 회귀, 1번 인덱스는 분류
            "y_reg": torch.tensor(self.y[i, 0], dtype=torch.float32),
            "y_cls": torch.tensor(int(self.y[i, 1]), dtype=torch.long),
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
    
    # 여기서 기존 "y"를 찾던 부분을 "y_reg"와 "y_cls"로 수정
    y_reg = stack_or_none("y_reg", torch.float32)
    y_cls = stack_or_none("y_cls", torch.long)
    
    # 반환도 6개로 맞춰줍니다
    return x_past, x_known, x_static, pos, y_reg, y_cls

# -------------------------
# Train / Eval
# -------------------------
def get_loss_fn(cfg: TFTConfig):
    if cfg.output_mode == "regression" or cfg.output_mode == "binary":
        return nn.MSELoss() if cfg.output_mode == "regression" else nn.BCELoss()
    if cfg.output_mode == "multiclass":
        return nn.CrossEntropyLoss()
    raise ValueError(cfg.output_mode)

# @torch.no_grad()
# def evaluate(model, loader, loss_fn, device):
#     model.eval()
#     total = 0.0
#     n = 0
#     for x_past, x_known, x_static, pos, y in loader:
#         x_past = x_past.to(device)
#         if x_known is not None: x_known = x_known.to(device)
#         if x_static is not None: x_static = x_static.to(device)
#         if pos is not None: pos = pos.to(device)
        
#         y = y.to(device)
#         y_hat, _ = model(x_past, x_known=x_known, x_static=x_static, pos=pos)
        
#         # multiclass: y should be (B,) long
#         if model.cfg.output_mode == "multiclass":
#             if y.dim() > 1:
#                 y = y.squeeze(-1)
#             y = y.long()
#         else:
#             y = y.float()
#             y_hat = y_hat.float()
            
#         loss = loss_fn(y_hat, y) if model.cfg.output_mode != "multiclass" else loss_fn(y_hat, y)
#         # criterion = MultiQuantileLoss(quantiles=[0.1, 0.5, 0.9]) # 260427 quantile_loss
#         # loss = criterion(y_hat.float(), y) # 260427 quantile_loss
#         total += float(loss.item()) * x_past.size(0)
        
#         n += x_past.size(0)
#     return total / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, loss_reg_fn, loss_cls_fn, device):
    model.eval()
    total = 0.0
    n = 0
    for x_past, x_known, x_static, pos, y_reg, y_cls in loader:
        x_past = x_past.to(device)
        if x_known is not None: x_known = x_known.to(device)
        if x_static is not None: x_static = x_static.to(device)
        if pos is not None: pos = pos.to(device)
        
        y_reg = y_reg.to(device)
        y_cls = y_cls.to(device)
        
        # Dual Output 수신
        (y_hat_reg, y_hat_cls), _ = model(x_past, x_known=x_known, x_static=x_static, pos=pos)
        
        # Hybrid Loss 계산 (train 루프와 동일한 비율 적용)
        loss_reg = loss_reg_fn(y_hat_reg.squeeze(), y_reg)
        loss_cls = loss_cls_fn(y_hat_cls, y_cls)
        loss = loss_reg + 0.01 * loss_cls
        
        total += float(loss.item()) * x_past.size(0)
        n += x_past.size(0)
        
    return total / max(n, 1)

@torch.no_grad()
def evaluate_lstm(model, loader, loss_fn, device):
    """LSTM 전용 평가 함수"""
    model.eval()
    total = 0.0
    n = 0
    # DataLoader가 반환하는 6개 인자 수신
    for x_past, x_known, x_static, pos, y_reg, y_cls in loader:
        x_past = x_past.to(device)
        y_reg = y_reg.to(device).float()
        
        y_hat, _ = model(x_past)
        # 차원 불일치(Broadcasting) 방지를 위해 squeeze 적용
        loss = loss_fn(y_hat.squeeze(), y_reg) 
        
        total += float(loss.item()) * x_past.size(0)
        n += x_past.size(0)
    return total / max(n, 1)

def train(
    cfg: TFTConfig,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    save_path: str = "tft_best.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = TemporalFusionTransformer(cfg).to(device)
    
    # loss_fn = get_loss_fn(cfg)
    # Hybrid Loss 세팅
    # loss_reg_fn = nn.HuberLoss(delta=2.0) # 이상치에 강한 Huber Loss 적용
    loss_reg_fn = MultiQuantileLoss(quantiles=cfg.quantiles)
    loss_cls_fn = nn.CrossEntropyLoss(label_smoothing=0.1) # 과적합 방지 및 Hit-Rate 학습

    # optim = torch.optim.Adam(model.parameters(), lr=lr)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4) # 260420 추가(과적합방지목적)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.2, patience=2, verbose=True)
    # Cosine Annealing 적용 (수렴 정체 돌파)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2, eta_min=1e-5)

    best = float("inf")
    bad = 0
    df_epoch = pd.DataFrame()

    for epoch in range(1, epochs + 1):
        model.train()
        for x_past, x_known, x_static, pos, y_reg, y_cls in train_loader:
            x_past = x_past.to(device)
            if x_known is not None: x_known = x_known.to(device)
            if x_static is not None: x_static = x_static.to(device)
            if pos is not None: pos = pos.to(device)
            # y = y.to(device)
            y_reg, y_cls = y_reg.to(device), y_cls.to(device)
            
            # y_hat, _ = model(x_past, x_known=x_known, x_static=x_static, pos=pos)
            # 모델의 Dual Output 수신
            (y_hat_reg, y_hat_cls), _ = model(x_past, x_known=x_known, x_static=x_static, pos=pos)
            
            # Loss 계산 및 합산 (람다 비중 조절 )
            loss_reg = loss_reg_fn(y_hat_reg.squeeze(), y_reg)
            loss_cls = loss_cls_fn(y_hat_cls, y_cls)
            loss = loss_reg + 0.01 * loss_cls

            # if cfg.output_mode == "multiclass":
            #     if y.dim() > 1:
            #         y = y.squeeze(-1)
            #     y = y.long()
            #     loss = loss_fn(y_hat, y)
            # else:
            #     y = y.float()
            #     loss = loss_fn(y_hat.float(), y)
            #     # criterion = MultiQuantileLoss(quantiles=[0.1, 0.5, 0.9]) # 260427 quantile_loss
                # loss = criterion(y_hat.float(), y) # 260427 quantile_loss
                
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
        scheduler.step()
        # 평가 진행 (새로운 evaluate 함수 사용)
        tr = evaluate(model, train_loader, loss_reg_fn, loss_cls_fn, device)
        val = evaluate(model, valid_loader, loss_reg_fn, loss_cls_fn, device)
        ts = evaluate(model, test_loader, loss_reg_fn, loss_cls_fn, device)
        
        # 기존 scheduler.step(val)은 삭제됨!
        
        print(f"***[TFT Epoch:{epoch:03d}] train_loss={tr:.6f} | val_loss={val:.6f} | test_loss={ts:.6f} | lr={optim.param_groups[0]['lr']:.2e}***")    
        
        # 모든 epoch의 모델 저장
        epoch_dir_tft = save_path.split('.')[0]
        os.makedirs(epoch_dir_tft, exist_ok=True) # 폴더가 없으면 자동 생성
        torch.save({"cfg": asdict(cfg), "state_dict": model.state_dict()}, epoch_dir_tft + f'/tft_{epoch}.pt')
        if val < best - 1e-8:
            best = val
            bad = 0
            torch.save({"cfg": asdict(cfg), "state_dict": model.state_dict()}, save_path)
            print(f" -> saved best to {save_path}")
        else:
            bad += 1
            if bad >= patience:
                print(" -> early stopping")
                break
        df_tmp = pd.DataFrame([{ 'epoch':epoch,
                                'train_loss':tr,
                                'val_loss':val,
                                'test_loss':ts,
                                'lr': optim.param_groups[0]['lr']
                              }])
        df_epoch = pd.concat([df_epoch, df_tmp])
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, df_epoch

# LSTM 학습용
def train_lstm(
    input_dim: int,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    hidden_dim: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    save_path: str = "lstm_best.pt"
):
    device = torch.device("cpu")
    model = SimpleLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1).to(device)
    
    loss_fn = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.2, patience=2)
    
    best = float("inf")
    bad = 0
    df_epoch = pd.DataFrame()
    
    for epoch in range(1, epochs + 1):
        model.train()
        # DataLoader가 반환하는 6개 인자 수신
        for x_past, x_known, x_static, pos, y_reg, y_cls in train_loader:
            x_past = x_past.to(device)
            y_reg = y_reg.to(device).float()
            
            y_hat, _ = model(x_past)
            # 차원 불일치 방지를 위해 squeeze 적용
            loss = loss_fn(y_hat.squeeze(), y_reg)
            
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            
        # LSTM 전용 평가 함수(evaluate_lstm) 호출
        tr = evaluate_lstm(model, train_loader, loss_fn, device)
        val = evaluate_lstm(model, valid_loader, loss_fn, device)
        ts = evaluate_lstm(model, test_loader, loss_fn, device)
        scheduler.step(val)
        
        print(f"***[LSTM Epoch:{epoch:03d}] train_loss={tr:.6f} | val_loss={val:.6f} | test_loss={ts:.6f} | lr={optim.param_groups[0]['lr']:.2e}***")
        
        epoch_dir_lstm = save_path.split('.')[0]
        os.makedirs(epoch_dir_lstm, exist_ok=True) # 폴더가 없으면 자동 생성
        torch.save(model.state_dict(), epoch_dir_lstm + f'/lstm_{epoch}.pt')
        if val < best - 1e-8:
            best = val
            bad = 0
            torch.save(model.state_dict(), save_path)
            print(f" -> saved best LSTM to {save_path}")
        else:
            bad += 1
            if bad >= patience:
                print(" -> early stopping")
                break
                
        df_tmp = pd.DataFrame([{ 'epoch':epoch,
                                'train_loss':tr,
                                'val_loss':val,
                                'test_loss':ts,
                                'lr': optim.param_groups[0]['lr']
                              }])
        df_epoch = pd.concat([df_epoch, df_tmp], ignore_index=True)
        
    # 최고 성능 가중치 로드
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, df_epoch

def main():
    print("참조\n (1) run_regression.py \n (2) prepare_data.py")
    raise SystemExit(
        "데이터 로드/전처리 부분(past/known/static) 작성후 들어감"
    )

if __name__ == "__main__":
    main()