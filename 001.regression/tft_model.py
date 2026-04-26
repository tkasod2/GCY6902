# -*- coding: utf-8 -*-
"""
Temporal Fusion Transformer (TFT) - Minimal PyTorch implementation (CPU-friendly)
- 목적: 산업재무 개별 항목(연속값) 또는 구간(분류/다중라벨)을 예측
- 기존 TF(T-LSTM) 코드의 입력 인터페이스(x, t, pos)를 최대한 유지하면서
  TFT 구조(변수선택+LSTM+Attention+Gating)를 적용할 수 있도록 설계
주의:
- 최소구현
- TODO부분 채울것
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Building blocks
# -----------------------------
class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """
    Gated Residual Network (TFT 핵심 블록)
    - 입력: x (.., in_dim)
    - context(optional): c (.., ctx_dim)  -> concat으로 반영
    - 출력: (.., out_dim)
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float = 0.1, ctx_dim: int = 0):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.fc1 = nn.Linear(in_dim + ctx_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(out_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if context is not None:
            x_in = torch.cat([x, context], dim=-1)
        else:
            x_in = x
        h = F.elu(self.fc1(x_in))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.glu(h)
        y = self.skip(x) + h
        return self.ln(y)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN)
    - 여러 변수(각각 embedding/projection된 벡터)를 가중합으로 결합
    - weights는 softmax로 산출
    입력:
      - var_inputs: (B, T, n_vars, d_model)  또는 (B, n_vars, d_model)도 가능
      - context(optional): (B, T, ctx_dim) 또는 (B, ctx_dim)
    출력:
      - combined: (B, T, d_model)  (또는 (B, d_model))
      - weights:  (B, T, n_vars)   (또는 (B, n_vars))
    """
    def __init__(self, n_vars: int, d_model: int, hidden_dim: int, dropout: float = 0.1, ctx_dim: int = 0):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.ctx_dim = ctx_dim
        # weights network: GRN -> n_vars logits
        self.weight_grn = GRN(in_dim=n_vars * d_model, out_dim=n_vars, hidden_dim=hidden_dim, dropout=dropout, ctx_dim=ctx_dim)
        # per-variable GRN
        self.var_grns = nn.ModuleList([GRN(d_model, d_model, hidden_dim, dropout=dropout) for _ in range(n_vars)])

    def forward(self, var_inputs: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # unify to (B, T, n_vars, d)
        if var_inputs.dim() == 3:
            var_inputs = var_inputs.unsqueeze(1)  # (B,1,n_vars,d)
            squeeze_time = True
        else:
            squeeze_time = False

        B, T, V, D = var_inputs.shape
        assert V == self.n_vars and D == self.d_model

        flat = var_inputs.reshape(B, T, V * D)

        if context is not None:
            if context.dim() == 2:
                context = context.unsqueeze(1)  # (B,1,ctx)
            w_logits = self.weight_grn(flat, context=context)
        else:
            w_logits = self.weight_grn(flat)

        weights = torch.softmax(w_logits, dim=-1)  # (B,T,V)

        # variable transformations
        transformed = []
        for i in range(V):
            transformed.append(self.var_grns[i](var_inputs[:, :, i, :]))  # (B,T,D)
        transformed = torch.stack(transformed, dim=2)  # (B,T,V,D)

        combined = (weights.unsqueeze(-1) * transformed).sum(dim=2)  # (B,T,D)

        if squeeze_time:
            combined = combined.squeeze(1)
            weights = weights.squeeze(1)
        return combined, weights


# -----------------------------
# TFT Model
# -----------------------------
@dataclass
class TFTConfig:
    d_model: int = 64
    hidden_dim: int = 128
    lstm_hidden: int = 128
    n_heads: int = 4
    dropout: float = 0.1

    # 입력 변수 개수(연속형 기준)
    # - past_vars: 과거 시점에서 관측되는 변수(재무 항목, 파생 피처 등)
    # - known_vars: 미래에도 미리 아는 변수(달력, 산업코드 고정, 금리/지표 시나리오 등)
    # - static_vars: 시계열에 고정된 변수(산업코드/기업규모/업종 등)
    past_vars: int = 10
    known_vars: int = 1   # 예: time index, month, quarter 등
    static_vars: int = 0

    output_dim: int = 1
    output_mode: str = "regression"  # "regression" | "binary" | "multiclass"
    n_classes: int = 0               # multiclass일 때만 사용


class TemporalFusionTransformer(nn.Module):
    """
    최소 TFT 구현
    입력 인터페이스(업무 편의):
      - x_past:  (B, L, past_vars)          과거 관측 변수
      - x_known: (B, L, known_vars)         known covariate (time, calendar 등)
      - x_static:(B, static_vars) or None   정적 변수
      - pos:     (B,) or None               기존 코드처럼 '특정 시점 hidden만 뽑기' 옵션
                 (pos=None이면 마지막 시점 output 사용)
    출력:
      - y_hat: (B, output_dim)  (pos로 1개 시점 선택 기준)
      - aux: dict (variable selection weights, attention weights 등)
    """
    def __init__(self, cfg: TFTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # 1) per-variable projection to d_model
        self.past_projs = nn.ModuleList([nn.Linear(1, d) for _ in range(cfg.past_vars)])
        self.known_projs = nn.ModuleList([nn.Linear(1, d) for _ in range(cfg.known_vars)]) if cfg.known_vars > 0 else nn.ModuleList()

        self.static_projs = nn.ModuleList([nn.Linear(1, d) for _ in range(cfg.static_vars)]) if cfg.static_vars > 0 else nn.ModuleList()

        # 2) static context encoders (optional)
        #    TFT는 static context를 여러 곳(GRN, VSN)에 넣는데, 여기서는 간단히 1개 context로 사용
        self.static_context_grn = GRN(in_dim=d, out_dim=d, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout) if cfg.static_vars > 0 else None

        # 3) Variable selection networks
        ctx_dim = d if cfg.static_vars > 0 else 0
        self.vsn_past = VariableSelectionNetwork(cfg.past_vars, d, cfg.hidden_dim, dropout=cfg.dropout, ctx_dim=ctx_dim)
        self.vsn_known = VariableSelectionNetwork(cfg.known_vars, d, cfg.hidden_dim, dropout=cfg.dropout, ctx_dim=ctx_dim) if cfg.known_vars > 0 else None

        # 4) LSTM encoder (TFT는 encoder/decoder 구분 가능하지만 여기선 단일 LSTM으로 단순화)
        self.lstm = nn.LSTM(input_size=d, hidden_size=cfg.lstm_hidden, num_layers=1, batch_first=True)
        self.lstm_proj = nn.Linear(cfg.lstm_hidden, d)

        # 5) Self-attention over time
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=cfg.n_heads, dropout=cfg.dropout, batch_first=True)

        # 6) Post-attention GRN + gating + layer norm
        self.post_attn_grn = GRN(in_dim=d, out_dim=d, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout, ctx_dim=0)

        # 7) Output head (time-distributed -> 선택 시점만 추출)
        self.out = nn.Linear(d, cfg.output_dim if cfg.output_mode != "multiclass" else cfg.n_classes)

    def _project_vars(self, x: torch.Tensor, projs: nn.ModuleList) -> torch.Tensor:
        """
        x: (B,L,V) or (B,V)
        return: (B,L,V,d) or (B,V,d)
        """
        if x.dim() == 2:
            B, V = x.shape
            outs = []
            for i in range(V):
                xi = x[:, i].unsqueeze(-1)  # (B,1)
                outs.append(projs[i](xi))   # (B,d)
            return torch.stack(outs, dim=1)  # (B,V,d)

        B, L, V = x.shape
        outs = []
        for i in range(V):
            xi = x[:, :, i].unsqueeze(-1)  # (B,L,1)
            outs.append(projs[i](xi))      # (B,L,d)
        return torch.stack(outs, dim=2)    # (B,L,V,d)

    def forward(
        self,
        x_past: torch.Tensor,
        x_known: Optional[torch.Tensor] = None,
        x_static: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # shapes
        B, L, Vp = x_past.shape
        assert Vp == self.cfg.past_vars

        # static context
        static_ctx = None
        static_weights = None
        if self.cfg.static_vars > 0:
            assert x_static is not None and x_static.shape[1] == self.cfg.static_vars
            s_emb = self._project_vars(x_static, self.static_projs)  # (B,Vs,d)
            # 간단히 평균 pooling
            s = s_emb.mean(dim=1)  # (B,d)
            static_ctx = self.static_context_grn(s)  # (B,d)
        # past var embeddings
        past_emb = self._project_vars(x_past, self.past_projs)  # (B,L,Vp,d)

        # known embeddings
        known_emb = None
        if self.cfg.known_vars > 0:
            assert x_known is not None and x_known.shape[2] == self.cfg.known_vars
            known_emb = self._project_vars(x_known, self.known_projs)  # (B,L,Vk,d)

        # Variable selection
        if static_ctx is not None:
            ctx_t = static_ctx.unsqueeze(1).expand(-1, L, -1)  # (B,L,d)
            past_sel, w_past = self.vsn_past(past_emb, context=ctx_t)
            if self.vsn_known is not None:
                known_sel, w_known = self.vsn_known(known_emb, context=ctx_t)
                x = past_sel + known_sel
            else:
                w_known = None
                x = past_sel
        else:
            past_sel, w_past = self.vsn_past(past_emb, context=None)
            if self.vsn_known is not None:
                known_sel, w_known = self.vsn_known(known_emb, context=None)
                x = past_sel + known_sel
            else:
                w_known = None
                x = past_sel

        # LSTM
        lstm_out, _ = self.lstm(x)           # (B,L,lstm_hidden)
        lstm_out = self.lstm_proj(lstm_out)  # (B,L,d)

        # Self-attention
        attn_out, attn_w = self.attn(lstm_out, lstm_out, lstm_out, need_weights=True)  # (B,L,d), (B,L,L)

        # Post-attention GRN
        y = self.post_attn_grn(attn_out)

        # choose timestep
        if pos is None:
            y_sel = y[:, -1, :]  # (B,d)
        else:
            # pos: (B,) with values in [0, L-1]
            idx = torch.arange(B, device=y.device)
            y_sel = y[idx, pos.long(), :]  # (B,d)

        out = self.out(y_sel)

        # output post-processing (기존 TF 코드 호환 옵션)
        if self.cfg.output_mode == "binary":
            out = torch.sigmoid(out)
        elif self.cfg.output_mode == "regression":
            # 필요시 output_dim==59일 때 일부만 sigmoid 적용하는 기존 룰을 사용할 수 있음
            out = out
        elif self.cfg.output_mode == "multiclass":
            # 학습 시 CrossEntropyLoss를 쓰면 여기서 softmax는 보통 생략
            pass
        else:
            raise ValueError(f"Unknown output_mode: {self.cfg.output_mode}")

        aux = {
            "w_past": w_past,                  # (B,L,Vp)
            "w_known": w_known if w_known is not None else torch.empty(0),
            "attn_w": attn_w,                  # (B,L,L)
            "static_weights": s_emb.mean(dim=1)
        }
        if static_ctx is not None:
            aux["static_ctx"] = static_ctx
        return out, aux

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.ModuleList([
            nn.LSTM(input_dim if i == 0 else hidden_dim, 
                    hidden_dim, batch_first=True) 
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, output_dim)
        from types import SimpleNamespace
        self.cfg = SimpleNamespace(output_mode='regression')

    def forward(self, x_past, **kwargs):
        out = x_past
        for lstm in self.lstm:
            out, _ = lstm(out)
        out = self.fc(out[:, -1, :])
        return out, {} # TFT와 출력 형식을 맞추기 위해 빈 dict 반환