# -*- coding: utf-8 -*-
import torch
import os
import gc
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader

# 기존 파일에서 클래스 및 함수 불러오기
from tft_model import TFTConfig, TemporalFusionTransformer
from train_tft import FinanceSeqDataset, collate_fn, train, train_lstm
from sklearn.metrics import mean_squared_error, mean_absolute_error

#sample test
from preprocess import *

def fit_and_out(df,
                target='Close',
                outdir = './output/',
                seq_length=30):
    past_vars = ['Open', 'High', 'Low', 'Close', 'Volume', 
                 'Quote asset volume', 'Number of trades', 
                 'Taker buy base asset volume', 'Taker buy quote asset volume']
    x_past, x_known, x_static, y, scaler, le = prepare_data(
        df,
        target=target,
        seq_length=seq_length
    )
    for i, name in enumerate(['Train', 'Valid', 'Test']):
        print(f"{name} set - x_past: {x_past[i].shape}, x_static: {x_static[i].shape}")

    df_scaler = pd.DataFrame({
        "VAR": scaler.feature_names_in_,
        "mean": scaler.mean_,
        "std": scaler.scale_
    })

    df_scaler.to_csv(outdir+"scaler.csv")

    train_ds = FinanceSeqDataset(x_past[0], y[0], x_known[0], x_static[0])
    valid_ds = FinanceSeqDataset(x_past[1], y[1], x_known[1], x_static[1])
    test_ds = FinanceSeqDataset(x_past[2], y[2], x_known[2], x_static[2])

    trn_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(valid_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    ts_loader = DataLoader(test_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)

    cfg = TFTConfig(
        d_model=64,
        hidden_dim=128,
        lstm_hidden=128,
        n_heads=4,
        dropout=0.2,

        past_vars=9,
        known_vars=2,
        static_vars=1,
        output_mode='regression'
    )
    print("=======1. TFT 모델 학습중=======")
    model = train(
        cfg = cfg,
        train_loader = trn_loader,
        valid_loader = val_loader,
        test_loader = ts_loader,
        epochs = 30, # 수정필요
        lr = 1e-3,
        patience = 5,
        save_path = outdir+f'{target}_tft_test_260312.pt'
    )
    print("=======2. LSTM 모델 학습중=======")
    model_lstm = train_lstm(
        input_dim=cfg.past_vars,
        train_loader=trn_loader,
        valid_loader=val_loader,
        test_loader=ts_loader,
        save_path=outdir + f'{target}_lstm_test_260312.pt'
    )

    print("====================output building 중====================")
    
    model.eval()
    model_lstm.eval()
    with torch.no_grad():

        all_actuals, all_tft_preds, all_lstm_preds, all_weights = [], [], [], []

        for x_p, x_k, x_s, p, y in val_loader:
            # tft
            y_tft, aux = model(x_p, x_known=x_k, x_static=x_s, pos=p)
            # lstm
            y_lstm, _ = model_lstm(x_p)

            all_actuals.append(y.numpy())
            all_tft_preds.append(y_tft.numpy())
            all_lstm_preds.append(y_lstm.numpy())
            all_weights.append(aux['w_past'].numpy())

    # 결과 data frame
    res_df = pd.DataFrame({
        'Actual': np.concatenate(all_actuals).flatten(),
        'TFT_Prediction': np.concatenate(all_tft_preds).flatten(),
        'LSTM_Prediction': np.concatenate(all_lstm_preds).flatten()
    })
    res_df.to_csv(outdir + f"{target}_comparison_results.csv", index=False)
    
    actual = res_df['Actual']
    tft_pred = res_df['TFT_Prediction']
    lstm_pred = res_df['LSTM_Prediction']

    tft_rmse = np.sqrt(mean_squared_error(actual, tft_pred))
    tft_mae = mean_absolute_error(actual, tft_pred)

    lstm_rmse = np.sqrt(mean_squared_error(actual, lstm_pred))
    lstm_mae = mean_absolute_error(actual, lstm_pred)

    df_pfmc = pd.DataFrame({'tft_rmse' : [tft_rmse],
                  'tft_mae' : [tft_mae],
                  'lstm_rmse' : [lstm_rmse],
                  'lstm_mae' : [lstm_mae]
                  })
    df_pfmc.to_csv(outdir + f"{target}_pfmc.csv", index=False)
    # tft 변수중요도 추출
    avg_weights = np.concatenate(all_weights).mean(axis=(0, 1))

    imp_df = pd.DataFrame({
        'Feature': past_vars,
        'Importance': avg_weights
    }).sort_values(by='Importance', ascending=False)
    imp_df.to_csv(outdir + f"{target}_tft_feature_importance.csv", index=False)
    
    # 시각화 차트 저장
    fig, axes = plt.subplots(1,2, figsize=(15,6))
    sns.scatterplot(data=res_df, x='Actual', y='TFT_Prediction'
                    , ax=axes[0], color='blue', alpha=0.6, markers='x',label='TFT')
    sns.scatterplot(data=res_df, x='Actual', y='LSTM_Prediction'
                    , ax=axes[0], color='grey', alpha=0.6, markers='x',label='LSTM')
    min_val = res_df['Actual'].min()
    max_val = res_df['Actual'].max()
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Identity')
    
    axes[0].set_title(f"Actual vs Prediction ({target})")
    axes[0].legend() # 범례 표시
    axes[0].grid(True, linestyle='--', alpha=0.5)

    sns.barplot(data=imp_df.head(10), x='Importance', y='Feature', ax=axes[1], palette='viridis')
    axes[1].set_title("TFT Top 10 Feature Importance")
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()

    save_path = outdir + f"{target}_comparison_summary.png"
    plt.savefig(save_path)
    print(f"Chart saved at: {save_path}")
    plt.show()

def load_and_output(df,
                    tft_path, 
                    lstm_path,
                target='Close',
                outdir = './output/',
                seq_length=30):
    past_vars = ['Open', 'High', 'Low', 'Close', 'Volume', 
                 'Quote asset volume', 'Number of trades', 
                 'Taker buy base asset volume', 'Taker buy quote asset volume']
    
    x_past, x_known, x_static, y, scaler, le = prepare_data(
        df,
        target=target,
        seq_length=seq_length
    )

    train_ds = FinanceSeqDataset(x_past[0], y[0], x_known[0], x_static[0])
    valid_ds = FinanceSeqDataset(x_past[1], y[1], x_known[1], x_static[1])
    test_ds = FinanceSeqDataset(x_past[2], y[2], x_known[2], x_static[2])

    trn_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(valid_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
    ts_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # 2. 모델 구조 선언 및 가중치 로드
    cfg = TFTConfig(
        d_model=64, hidden_dim=128, lstm_hidden=128, n_heads=4,
        dropout=0.2, past_vars=9, known_vars=2, static_vars=1, output_mode='regression'
    )

    print("====================output building 중====================")
    # model load
    model = TemporalFusionTransformer(cfg)
    checkpoint = torch.load(tft_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    # LSTM 로드
    from tft_model import SimpleLSTM
    model_lstm = SimpleLSTM(input_dim=cfg.past_vars, hidden_dim=128, output_dim=1)
    model_lstm.load_state_dict(torch.load(lstm_path, map_location='cpu'))

    model.eval()
    model_lstm.eval()
    with torch.no_grad():

        all_actuals, all_tft_preds, all_lstm_preds, all_weights = [], [], [], []

        for x_p, x_k, x_s, p, y in ts_loader:
            # tft
            y_tft, aux = model(x_p, x_known=x_k, x_static=x_s, pos=p)
            # lstm
            y_lstm, _ = model_lstm(x_p)

            all_actuals.append(y.numpy())
            all_tft_preds.append(y_tft.numpy())
            all_lstm_preds.append(y_lstm.numpy())
            all_weights.append(aux['w_past'].numpy())

    # 결과 data frame
    res_df = pd.DataFrame({
        'Actual': np.concatenate(all_actuals).flatten(),
        'TFT_Prediction': np.concatenate(all_tft_preds).flatten(),
        'LSTM_Prediction': np.concatenate(all_lstm_preds).flatten()
    })
    res_df.to_csv(outdir + f"{target}_comparison_results.csv", index=False)
    
    actual = res_df['Actual']
    tft_pred = res_df['TFT_Prediction']
    lstm_pred = res_df['LSTM_Prediction']

    tft_rmse = np.sqrt(mean_squared_error(actual, tft_pred))
    tft_mae = mean_absolute_error(actual, tft_pred)

    lstm_rmse = np.sqrt(mean_squared_error(actual, lstm_pred))
    lstm_mae = mean_absolute_error(actual, lstm_pred)

    df_pfmc = pd.DataFrame({'tft_rmse' : [tft_rmse],
                  'tft_mae' : [tft_mae],
                  'lstm_rmse' : [lstm_rmse],
                  'lstm_mae' : [lstm_mae]
                  })
    df_pfmc.to_csv(outdir + f"{target}_pfmc.csv", index=False)
    # tft 변수중요도 추출
    avg_weights = np.concatenate(all_weights).mean(axis=(0, 1))
    imp_df = pd.DataFrame({
        'Feature': past_vars,
        'Importance': avg_weights
    }).sort_values(by='Importance', ascending=False)
    imp_df.to_csv(outdir + f"{target}_tft_feature_importance.csv", index=False)
    
    # 시각화 차트 저장
    fig, axes = plt.subplots(1,2, figsize=(15,6))
    sns.scatterplot(data=res_df, x='Actual', y='TFT_Prediction'
                    , ax=axes[0], color='blue', alpha=0.6, markers='x',label='TFT')
    sns.scatterplot(data=res_df, x='Actual', y='LSTM_Prediction'
                    , ax=axes[0], color='grey', alpha=0.6, markers='x',label='LSTM')
    min_val = res_df['Actual'].min()
    max_val = res_df['Actual'].max()
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Identity')
    
    axes[0].set_title(f"Actual vs Prediction ({target})")
    axes[0].legend() # 범례 표시
    axes[0].grid(True, linestyle='--', alpha=0.5)

    sns.barplot(data=imp_df.head(10), x='Importance', y='Feature', ax=axes[1], palette='viridis')
    axes[1].set_title("TFT Top 10 Feature Importance")
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()

    save_path = outdir + f"{target}_comparison_summary.png"
    plt.savefig(save_path)
    print(f"Chart saved at: {save_path}")
    plt.show()