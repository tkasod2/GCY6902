# prepare_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import gc

def prepare_data(df_in: pd.DataFrame,
                 target,
                 var_list,
                 trk_start_dt='20240101',  # 시작 일자
                 seq_length=60, # 60개월
                 dt_cut='20251231',         # train/valid 분할 기준 일자
                 grp_cd = 'Symbol',
                 tgt_gap = 1):
    """
    df_in : Binance 데이터
    target : 추정하고자하는 Binance 항목
    trk_start_dt : 보유정보 최초의 관찰 시작 일자
    seq_length : n개월 데이터로 추정할것인지에 대한 params
    dt_cut : train/valid 데이터의 max 일자
    grp_cd : 산업그룹 코드
    tgt_gap : 예측하는 시점의 시차
        ex) tgt_gap = 0 : 2024년 9월까지의 데이터로 2024년 10월 정보를 예측
            tgt_gap = 3 : 2024년 9월까지의 데이터로 2025년 01월 정보를 예측
    """

    # # 1. Data 세팅
    # # (1) 일반적인 공변량 : Binance
    #
    past_vars = var_list
    df = df_in.loc[df_in['BAS_DT'] >= trk_start_dt, :].copy()
    df = df.sort_values(by=[grp_cd, 'BAS_DT']).reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(df.loc[df['BAS_DT'] <= dt_cut, past_vars]) # train data로 fitting

    raw_target_series = df[target].copy()

    df[past_vars] = scaler.transform(df[past_vars]) # 전체 데이터에 대해서 scaling 진행
    
    #target 순서 확인
    dct_target = {col: idx for idx, col in enumerate(past_vars)}
    ynum = dct_target[target]

    target_scalers = {}
    for g_name, group_idx in df.groupby(grp_cd).groups.items():
        g_scaler = StandardScaler()

        # 해당 그룹의 Train 기간 데이터로만 fit
        train_mask = (df.loc[group_idx, 'BAS_DT'] <= dt_cut)
        train_target_raw = raw_target_series.loc[group_idx][train_mask].values.reshape(-1, 1)

        if len(train_target_raw) > 0:
            g_scaler.fit(train_target_raw)
        else:
            # Train 데이터가 극도로 부족한 예외 그룹은 전체 평균 사용
            g_scaler.mean_ = np.array([scaler.mean_[ynum]])
            g_scaler.scale_ = np.array([scaler.scale_[ynum]])
            
        target_scalers[g_name] = g_scaler
        
        # 해당 그룹의 전체 target 데이터를 개별 transform하여 덮어쓰기
        raw_group_all = raw_target_series.loc[group_idx].values.reshape(-1, 1)
        df.loc[group_idx, target] = g_scaler.transform(raw_group_all).flatten()
    # ----------------------------------------------------

    df[past_vars] = df[past_vars].fillna(0) # 결측치 보간은 평균값 대치(scaling정보이므로 0이 평균값)

    # # (2) Known 공변량 : 연도 index
    df['tmp_dt'] = pd.to_datetime(df['BAS_DT'].astype(str), format='%Y%m%d')
    base_dt = pd.to_datetime(str(trk_start_dt), format='%Y%m%d')
    
    # 두 날짜의 차이를 구하고 일(days) 단위로 변환
    df['dt_idx'] = (df['tmp_dt'] - base_dt).dt.days
    
    del df['tmp_dt']
    known_vars = ['dt_idx']
    
    # # (3) Static 공변량 : 산업분류코드
    le = LabelEncoder()
    le.fit(df[grp_cd])
    df['grp_cd_idx'] = le.transform(df[grp_cd])
    static_vars = ['grp_cd_idx']

    # # (4) target (부도율)
    lst_final_cols = ['BAS_DT', grp_cd, target] + var_list + known_vars + static_vars
    lst_final_cols = list(dict.fromkeys(lst_final_cols))
    df = df[lst_final_cols].copy()
    gc.collect()


    print(f"====target : {ynum+1} {target}=====")

    # # (5) 시퀀스 생성
    # # 260404 수정내용
    def shift_dt(dt_val, days):
        """정수형 YYYYMMDD를 받아서 days만큼 이동시킨 후 다시 정수형 YYYYMMDD 반환"""
        dt = pd.to_datetime(str(dt_val), format='%Y%m%d')
        shifted = dt + pd.Timedelta(days=days) # 일(day) 단위 시프팅
        return int(shifted.strftime('%Y%m%d'))

    def make_sequence(df_in, tvt=1):
        df = df_in.copy()
        test_valid_period = 60
        train_end_dt = shift_dt(dt_cut, -test_valid_period)              # dt cut 기준으로 60일 전까지를 train set
        valid_start_dt = shift_dt(shift_dt(dt_cut, 30), -test_valid_period-seq_length-tgt_gap)
        # dt cut 기준으로 5년 전부터 valid set (시퀀스 길이 확보용)
        end_dt = df['BAS_DT'].max()
        test_start_dt = shift_dt(shift_dt(end_dt, 30), -test_valid_period-seq_length-tgt_gap)
        # max BAS_DT 기준으로 1년 기간 test set (시퀀스 길이 확보용)

        if tvt == 1: # train
            df = df.loc[df['BAS_DT'] <= str(train_end_dt), :].copy()
            print(f"train: {trk_start_dt}~{train_end_dt} 자료 추출 중 : {len(df)}")

        elif tvt == 2: # valid
            df = df.loc[(df['BAS_DT'] >= str(valid_start_dt)) & (df['BAS_DT'] <= str(dt_cut)), :].copy()
            print(f"validation set : {valid_start_dt}~{dt_cut} 자료 추출 중 : {len(df)}")

        else: # test
            df = df.loc[df['BAS_DT'] >= str(test_start_dt), :].copy()
            print(f"test set : {test_start_dt}~{end_dt} 자료 추출 중 : {len(df)}")

        # df_raw = df.copy()
        # df_raw[past_vars] = scaler.inverse_transform(df_raw[past_vars])

        x_past_list, x_known_list, x_static_list, y_list = [], [], [], []
        # 역변환을 위한 scaler 정보 (타겟 변수의 mean과 std) # 🎆🎇260501

        for g_name, group in df.groupby(grp_cd, sort=False):
            if len(group) < seq_length: continue
            # 해당 그룹 전용 mean, std 추출
            mean_y = target_scalers[g_name].mean_[0]
            std_y = target_scalers[g_name].scale_[0]

            group_past = group[past_vars].values
            group_known = group[known_vars].values.reshape(-1, 1)
            group_static = group[static_vars].iloc[0]

            #
            for i in range(len(group) - seq_length - tgt_gap):
                x_past_list.append(group_past[i:i+seq_length])
                x_known_list.append(group_known[i:i+seq_length]) # 시계열 정보는 아는정보이니까
                x_static_list.append(group_static)

                # 타겟 수치
                y_val = group_past[i+seq_length+tgt_gap, ynum] # 🎆🎇260501
                # 방향성(Hit-Rate) 클래스 계산을 위한 기준값(t시점)과 미래값(t+gap) 역변환 # 🎆🎇260501
                y_base_scaled = group_past[i+seq_length-1, ynum] # 🎆🎇260501
                raw_base = y_base_scaled * std_y + mean_y # 🎆🎇260501
                raw_target = y_val * std_y + mean_y # 🎆🎇260501

                # 상대 변동률 계산 # 🎆🎇260501
                diff = (raw_target - raw_base) / (raw_base + 1e-9) 
                
                if diff > 0.05:
                    y_cls = 1 # 상승 (악화)
                elif diff < -0.05:
                    y_cls = 2 # 하락 (개선)
                else:
                    y_cls = 0 # 유지 (±5% 이내)
                    
                # y_list에 회귀값과 분류클래스를 쌍으로 저장
                y_list.append([y_val, y_cls])

                # y_list.append(group_past[i+seq_length+tgt_gap, ynum])

        x_past = np.array(x_past_list, np.float32)
        x_known = np.array(x_known_list, int)
        x_static = np.array(x_static_list, int)
        
        y = np.array(y_list, np.float32)
        # y = np.array(y_list, np.float32).reshape(-1, 1)
        return x_past, x_known, x_static, y

    x_past_1, x_known_1, x_static_1, y_1 = make_sequence(df, 1)
    x_past_2, x_known_2, x_static_2, y_2 = make_sequence(df, 2)
    x_past_3, x_known_3, x_static_3, y_3 = make_sequence(df, 3)

    x_past = [x_past_1, x_past_2, x_past_3]
    x_known = [x_known_1, x_known_2, x_known_3]
    x_static = [x_static_1, x_static_2, x_static_3]
    y = [y_1, y_2, y_3]

    return x_past, x_known, x_static, y, scaler, le, target_scalers