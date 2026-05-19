import torch
import os
import pickle
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from models.dl.tft_model import TFTConfig, TemporalFusionTransformer

class TFTDirectTrajectoryEvaluator:
    def __init__(self, model_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        
        # 아티팩트 로드
        with open(os.path.join(model_dir, "config.json"), "r", encoding='utf-8') as f:
            config_dict = json.load(f)
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

    def evaluate_weekly_strategy(
        self, 
        df_raw, 
        seq_length, 
        base_dt_str='2026040500', 
        target_dt_str='2026041200', 
        trk_start_dt='2024030100', 
        threshold=0.03,
        tgt_gap=20,             # 학습 시 설정한 타겟 갭 (20스텝)
        target='Close'
    ):
        """
        [반환값 아키텍처 개편]
        Returns:
            report_df: 기존의 심볼당 1개 행으로 마스터 정산된 전략 요약 테이블
            trajectory_df: Streamlit 시각화용 (1)Symbol + (3)타임스탬프별 예상 가격 및 실제 가격 롱포맷 테이블
            xai_dict: (2)Symbol별 변수 중요도(Feature Importance) 데이터프레임 맵
        """
        results = []
        trajectory_records = [] 
        xai_dict = {}
        
        var_list = list(self.scaler.feature_names_in_)
        
        df = df_raw.copy()
        df = df.sort_values(['Symbol','BAS_DT'])
        df[var_list] = df.groupby('Symbol')[var_list].ffill() # 결측치는 동일 심볼 내 이전값 활용

        df['BAS_DT'] = df['BAS_DT'].astype(str).str.zfill(10)
        df['tmp_dt'] = pd.to_datetime(df['BAS_DT'], format='%Y%m%d%H')
        anchor_dt = pd.to_datetime(str(trk_start_dt).zfill(10), format='%Y%m%d%H')
        
        # 8시간 단위 타임스텝 인덱스 생성
        df['dt_idx'] = (df['tmp_dt'] - anchor_dt) // pd.Timedelta(hours=8)
        
        all_symbols = df['Symbol'].unique()
        
        for symbol in tqdm(all_symbols, desc="Evaluating Symbols"):

            if symbol not in self.le.classes_: continue
            g_scaler = self.target_scalers[symbol]
            
            start_row = df[(df['Symbol'] == symbol) & (df['BAS_DT'] == base_dt_str)]
            if start_row.empty: continue
            start_price = start_row.iloc[0][target]
            
            target_timeline = df[(df['Symbol'] == symbol) &
                                    (df['BAS_DT'] >= base_dt_str) &
                                    (df['BAS_DT'] <= target_dt_str)].copy()
            target_timeline = target_timeline.sort_values('BAS_DT')
            
            future_dates = list(target_timeline['dt_idx'].unique())
            
            pred_trajectory = []   
            actual_trajectory = [] 
            symbol_vsn_weights = [] 
            
            # 2. 21개 시점 각각 독립 예측 실행
            for idx, f_date in enumerate(future_dates):
                
                tgt_row = df[(df['Symbol'] == symbol) & (df['dt_idx'] == f_date)]
                
                if tgt_row.empty:
                    pred_trajectory.append(np.nan)
                    actual_trajectory.append(np.nan)
                    continue
                # 현재 추정 타겟 시점의 dt_idx 확보
                f_date_idx = tgt_row.iloc[0]['dt_idx']
                act_price = tgt_row.iloc[0][target]
                
                group_history = df[
                    (df['Symbol'] == symbol) & 
                    (df['dt_idx'] <= f_date_idx) & 
                    (df['dt_idx'] >= f_date_idx - tgt_gap - seq_length)
                ].copy()
                group_history = group_history.sort_values('dt_idx')
                
                if len(group_history) < seq_length: 
                    pred_trajectory.append(np.nan)
                    actual_trajectory.append(np.nan)
                    continue
                
                tgt_row = df[(df['Symbol']==symbol)&
                                (df['dt_idx']==f_date)
                                ]
                input_seq = group_history.head(seq_length).copy()
                actual_trajectory.append(act_price)
                # 스케일링 진행
                target_raw = input_seq[target].copy()
                input_seq[var_list] = self.scaler.transform(input_seq[var_list])
                input_seq[target] = target_raw.copy()

                input_seq[target] = g_scaler.transform(
                    input_seq[[target]].values
                ).flatten()
                x_past = torch.tensor(input_seq[var_list].values, dtype=torch.float32).unsqueeze(0).to(self.device)
                x_known = torch.tensor(input_seq[['dt_idx']].values, dtype=torch.float32).unsqueeze(0).to(self.device)
                x_static = torch.tensor([[self.le.transform([symbol])[0]]], dtype=torch.float32).to(self.device)
                
                with torch.no_grad():
                    y_tuple, aux = self.model(x_past, x_known=x_known, x_static=x_static)
                    q50_scaled = y_tuple[0][0, 1].cpu().item()
                    symbol_vsn_weights.append(aux['w_past'].cpu().numpy())
                    
                pred_price = q50_scaled * g_scaler.scale_[0] + g_scaler.mean_[0]
                pred_trajectory.append(pred_price)
                
                trajectory_records.append({
                    'Symbol': symbol,
                    'Step': f"T+{idx+1}",
                    'BAS_DT': f_date,
                    'Predicted_Price': pred_price,
                    'Actual_Price': act_price,
                    'Recent_Price': target_raw.iloc[-1]
                })
            
            pred_trajectory = np.array(pred_trajectory)
            actual_trajectory = np.array(actual_trajectory)
            
            if np.isnan(pred_trajectory).any(): 
                continue
            
            # (2) VSN 중요도 가중치 산출 및 맵 저장
            avg_vsn = np.concatenate(symbol_vsn_weights, axis=0).mean(axis=(0, 1))
            df_imp = pd.DataFrame({'Feature': var_list, 'Importance': avg_vsn}).sort_values('Importance', ascending=False).reset_index(drop=True)
            xai_dict[symbol] = df_imp
            
            # 3. 21개 범위 안에서 수익률 최대화 매매 타이밍 산출
            # 1) Long
            best_buy_step = np.argmin(pred_trajectory)
            if best_buy_step < len(pred_trajectory) - 1:
                best_sell_step = np.argmax(pred_trajectory[best_buy_step:]) + best_buy_step
            else:
                best_sell_step = best_buy_step
            
            pred_buy_price = pred_trajectory[best_buy_step]
            pred_sell_price = pred_trajectory[best_sell_step]
            long_return = (pred_sell_price - pred_buy_price) / (pred_buy_price + 1e-9)

            # 2) Short
            best_short_step = np.argmax(pred_trajectory)
            if best_short_step < len(pred_trajectory) - 1:
                best_cover_step = np.argmin(pred_trajectory[best_short_step:]) + best_short_step
            else:
                best_cover_step = best_short_step
            pred_short_price = pred_trajectory[best_short_step]
            pred_cover_price = pred_trajectory[best_cover_step]
            short_return = (pred_short_price - pred_cover_price) / (pred_short_price + 1e-9) # 방향 반대

            if long_return >= short_return:
                final_direction = 'LONG'
                expected_max_return = long_return
                entry_step, exit_step = best_buy_step, best_sell_step
            else:
                final_direction = 'SHORT'
                expected_max_return = short_return
                entry_step, exit_step = best_short_step, best_cover_step

            # expected_max_return = (pred_sell_price - pred_buy_price) / (pred_buy_price + 1e-9)
            
            # actual_buy_price = actual_trajectory[best_buy_step]
            # actual_sell_price = actual_trajectory[best_sell_step]
            # actual_strategy_return = (actual_sell_price - actual_buy_price) / (actual_buy_price + 1e-9)
            passive_actual_return = (actual_trajectory[-1] - start_price) / (start_price + 1e-9)

            if not np.isnan(actual_trajectory).any():
                act_entry_price = actual_trajectory[entry_step]
                act_exit_price = actual_trajectory[exit_step]
                
                if final_direction == 'LONG':
                    actual_strategy_return = (act_exit_price - act_entry_price) / (act_entry_price + 1e-9)
                else: # SHORT
                    actual_strategy_return = (act_entry_price - act_exit_price) / (act_entry_price + 1e-9)
            else:
                actual_strategy_return = np.nan
            # ----------------------------------------------------
            
            results.append({
                'Symbol': symbol,
                'Current_Price(4/5)': start_price,
                'Predicted_Entry_Price': pred_trajectory[entry_step],
                'Predicted_Exit_Price': pred_trajectory[exit_step],
                'Best_Entry_Timing': f"T+{entry_step+1}",   
                'Best_Exit_Timing': f"T+{exit_step+1}", 
                'Expected_Return_Pct': expected_max_return * 100,
                'Actual_Strategy_Return_Pct': actual_strategy_return * 100 if not np.isnan(actual_strategy_return) else np.nan,
                'Passive_Market_Return_Pct': passive_actual_return*100, 
                'Strategy_Signal': final_direction if expected_max_return > threshold else 'HOLD'
            })

                
        report_df = pd.DataFrame(results)
        trajectory_df = pd.DataFrame(trajectory_records)
        
        if not report_df.empty:
            report_df = report_df.sort_values('Expected_Return_Pct', ascending=False).reset_index(drop=True)
            
        return report_df, trajectory_df, xai_dict