import torch
import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from models.dl.tft_model import TFTConfig, TemporalFusionTransformer

class TFTDirectTrajectoryEvaluator:
    def __init__(self, model_dir, config_dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        
        # 아티팩트 로드
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            self.le = pickle.load(f)
        with open(os.path.join(model_dir, 'target_scalers.pkl'), 'rb') as f:
            self.target_scalers = pickle.load(f)
            
        # 모델 빌드
        self.cfg = TFTConfig(**config_dict)
        self.model = TemporalFusionTransformer(self.cfg).to(self.device)
        
        ckpt_path = os.path.join(model_dir, 'tft_best.pt')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
            print("✅ Direct Inference Engine 로드 완료")
        else:
            raise FileNotFoundError(f"❌ 가중치 파일 없음: {ckpt_path}")
            
        self.model.eval()

    def evaluate_weekly_strategy(self, df_raw, seq_length, base_dt_str='2026040500', target_dt_str='2026041200', trk_start_dt='2024030100', threshold=0.03):
        """
        [반환값 아키텍처 개편]
        Returns:
            report_df: 기존의 심볼당 1개 행으로 마스터 정산된 전략 요약 테이블
            trajectory_df: Streamlit 시각화용 (1)Symbol + (3)타임스탬프별 예상 가격 및 실제 가격 롱포맷 테이블
            xai_dict: (2)Symbol별 변수 중요도(Feature Importance) 데이터프레임 맵
        """
        results = []
        trajectory_records = [] # (1)과 (3)을 결합하여 시각화용 데이터를 적재할 리스트
        xai_dict = {}           # (2) XAI 중요도를 저장할 딕셔너리
        
        var_list = list(self.scaler.feature_names_in_)
        
        df = df_raw.copy()
        df['BAS_DT'] = df['BAS_DT'].astype(str).str.zfill(10)
        df['tmp_dt'] = pd.to_datetime(df['BAS_DT'], format='%Y%m%d%H')
        anchor_dt = pd.to_datetime(str(trk_start_dt).zfill(10), format='%Y%m%d%H')
        df['dt_idx'] = ((df['tmp_dt'] - anchor_dt).dt.total_seconds() // (3600 * 8)).astype(int)
        
        all_symbols = df['Symbol'].unique()
        
        for symbol in tqdm(all_symbols, desc="Evaluating Symbols"):
            try:
                if symbol not in self.le.classes_: continue
                g_scaler = self.target_scalers[symbol]
                
                # 1. 진입 시점(4/5 00시) 실제 가격 확보
                start_row = df[(df['Symbol'] == symbol) & (df['BAS_DT'] == base_dt_str)]
                if start_row.empty: continue
                start_price = start_row.iloc[0]['Close']
                
                # 해당 심볼의 미래 데이터 중 정확히 '일주일치(21개 스탬프)'만 슬라이싱하여 고정
                target_timeline = df[(df['Symbol'] == symbol) & (df['BAS_DT'] > base_dt_str) & (df['BAS_DT'] <= target_dt_str)].sort_values('BAS_DT')
                if len(target_timeline) < 21: continue # 데이터가 부족하면 제외
                
                target_timeline = target_timeline.head(21) # 정확히 21개 스탬프로 리미트 조절
                future_dates = target_timeline['BAS_DT'].values
                
                pred_trajectory = []   # 21개 추정치 저장
                actual_trajectory = [] # 21개 실제 정답 저장
                symbol_vsn_weights = [] # (2) XAI 취합용 리스트
                
                # 2. 21개 시점 각각 독립 예측 실행
                for idx, f_date in enumerate(future_dates):
                    group_history = df[(df['Symbol'] == symbol) & (df['BAS_DT'] < f_date)].sort_values('BAS_DT')
                    if len(group_history) < seq_length: 
                        pred_trajectory.append(np.nan)
                        actual_trajectory.append(np.nan)
                        continue
                    
                    input_seq = group_history.tail(seq_length).copy()
                    
                    tgt_row = df[(df['Symbol'] == symbol) & (df['BAS_DT'] == f_date)]
                    act_price = tgt_row.iloc[0]['Close'] if not tgt_row.empty else np.nan
                    actual_trajectory.append(act_price)
                    
                    input_seq[var_list] = self.scaler.transform(input_seq[var_list])
                    input_seq['Close'] = g_scaler.transform(input_seq[['Close']].values).flatten()
                    
                    x_past = torch.tensor(input_seq[var_list].values, dtype=torch.float32).unsqueeze(0).to(self.device)
                    x_known = torch.tensor(input_seq[['dt_idx']].values, dtype=torch.float32).unsqueeze(0).to(self.device)
                    x_static = torch.tensor([[self.le.transform([symbol])[0]]], dtype=torch.float32).to(self.device)
                    
                    with torch.no_grad():
                        y_tuple, aux = self.model(x_past, x_known=x_known, x_static=x_static)
                        q50_scaled = y_tuple[0][0, 1].cpu().item()
                        # (2) XAI 산출을 위한 변수별 중요도 가중치 수집
                        symbol_vsn_weights.append(aux['w_past'].cpu().numpy())
                        
                    pred_price = q50_scaled * g_scaler.scale_[0] + g_scaler.mean_[0]
                    pred_trajectory.append(pred_price)
                    
                    # (3) 타임스탬프 단위의 상세 시각화 레코드 적재
                    trajectory_records.append({
                        'Symbol': symbol,
                        'Step': f"T+{idx+1}",
                        'BAS_DT': f_date,
                        'Predicted_Price': pred_price,
                        'Actual_Price': act_price
                    })
                
                pred_trajectory = np.array(pred_trajectory)
                actual_trajectory = np.array(actual_trajectory)
                
                if np.isnan(pred_trajectory).any() or np.isnan(actual_trajectory).any(): continue
                
                # (2) VSN 중요도 가중치 산출 및 맵 저장 (run_regression.py 수식 전면 준수)
                avg_vsn = np.concatenate(symbol_vsn_weights, axis=0).mean(axis=(0, 1))
                df_imp = pd.DataFrame({'Feature': var_list, 'Importance': avg_vsn}).sort_values('Importance', ascending=False).reset_index(drop=True)
                xai_dict[symbol] = df_imp
                
                # 3. 21개 범위 안에서 수익률 최대화 매매 타이밍 산출
                best_buy_step = np.argmin(pred_trajectory)
                if best_buy_step < len(pred_trajectory) - 1:
                    best_sell_step = np.argmax(pred_trajectory[best_buy_step:]) + best_buy_step
                else:
                    best_sell_step = best_buy_step
                
                pred_buy_price = pred_trajectory[best_buy_step]
                pred_sell_price = pred_trajectory[best_sell_step]
                
                # 예상 및 실제 전략 수익률 계산
                expected_max_return = (pred_sell_price - pred_buy_price) / (pred_buy_price + 1e-9)
                actual_buy_price = actual_trajectory[best_buy_step]
                actual_sell_price = actual_trajectory[best_sell_step]
                actual_strategy_return = (actual_sell_price - actual_buy_price) / (actual_buy_price + 1e-9)
                
                # 단순 보유(Buy & Hold) 수익률
                passive_actual_return = (actual_trajectory[-1] - start_price) / (start_price + 1e-9)
                
                # 기존 전략 요약 아웃풋 스펙 그대로 유지
                results.append({
                    'Symbol': symbol,
                    'Current_Price(4/5)': start_price,
                    'Predicted_Min_Price': pred_buy_price,
                    'Predicted_Max_Price': pred_sell_price,
                    'Best_Buy_Timing': f"T+{best_buy_step+1}",   
                    'Best_Sell_Timing': f"T+{best_sell_step+1}", 
                    'Expected_Return_Pct': expected_max_return * 100,
                    'Actual_Strategy_Return_Pct': actual_strategy_return * 100,
                    'Passive_Market_Return_Pct': passive_actual_return * 100,
                    'Strategy_Signal': 'BUY' if expected_max_return > threshold else 'HOLD'
                })
            except Exception as e:
                continue
                
        report_df = pd.DataFrame(results)
        trajectory_df = pd.DataFrame(trajectory_records)
        
        if not report_df.empty:
            report_df = report_df.sort_values('Expected_Return_Pct', ascending=False).reset_index(drop=True)
            
        return report_df, trajectory_df, xai_dict