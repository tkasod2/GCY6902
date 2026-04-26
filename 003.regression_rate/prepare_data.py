import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import gc

def prepare_data(df_in: pd.DataFrame,
                 target,
                 var_list,
                 trk_start_dt='2024-01-01',  # 시작 일자
                 seq_length=60,               # 60일 데이터로 추정
                 dt_cut='2025-12-31',         # train/valid 분할 기준 일자
                 grp_cd = 'Symbol',           # 그룹 코드 (예: 심볼)
                 output_mode = 'regression'):
    
    """
    df_in : 시계열 데이터 (일단위)
    target : 추정하고자 하는 항목
    trk_start_dt : 데이터 관찰 시작일 (YYYY-MM-DD)
    seq_length : n일 시퀀스로 추정할 것인지
    dt_cut : 학습 데이터의 마지노선 일자
    """

    # 1. 날짜 형식 변환
    df = df_in.copy()
    df['BAS_DT'] = pd.to_datetime(df['Close time'])
    trk_start_dt = pd.to_datetime(trk_start_dt)
    dt_cut = pd.to_datetime(dt_cut)

    # 2. Data 세팅 및 정렬
    past_vars = var_list
    df = df.loc[df['BAS_DT'] >= trk_start_dt, :].copy()
    df = df.sort_values(by=[grp_cd, 'BAS_DT']).reset_index(drop=True)

    # 3. Scaling
    scaler = StandardScaler()
    # train 데이터(dt_cut 이전)로 피팅
    train_mask = df['BAS_DT'] <= dt_cut
    scaler.fit(df.loc[train_mask, past_vars])
    df[past_vars] = scaler.transform(df[past_vars])
    
    # 결측치 보간
    df[past_vars] = df[past_vars].fillna(0)

    # 4. Known 공변량 : 일자 index (시작일로부터 며칠째인지)
    df['dt_idx'] = (df['BAS_DT'] - trk_start_dt).dt.days
    known_vars = ['dt_idx']

    # 5. Static 공변량 : 그룹/심볼 인코딩
    le = LabelEncoder()
    df['grp_idx'] = le.fit_transform(df[grp_cd])
    static_vars = ['grp_idx']

    # 컬럼 정리
    lst_final_cols = ['BAS_DT', grp_cd, target] + var_list + known_vars + static_vars
    lst_final_cols = list(dict.fromkeys(lst_final_cols))
    df = df[lst_final_cols].copy()
    gc.collect()

    # Target 위치 인덱스 확인
    dct_target = {col: idx for idx, col in enumerate(past_vars)}
    ynum = dct_target[target]
    print(f"=====target : {ynum+1} {target}=====")

    # 6. 시퀀스 생성 함수 정의
    def shift_dt(dt, days):
        """datetime 객체를 받아서 days만큼 이동"""
        return dt + pd.DateOffset(days=days)
    
    def make_sequence(df_in, tvt=1):
        df_sub = df_in.copy()
        
        # 날짜 구간 설정 (일단위 기준)
        # 예: 학습 데이터는 기준일 30일 전까지, 검증 데이터는 기준일 전후 등 (사용자 목적에 맞춰 조정 가능)
        train_end_dt = shift_dt(dt_cut, -14)            # 기준일 2주 전까지 train
        valid_start_dt = shift_dt(dt_cut, -(seq_length + 30)) # 시퀀스 길이를 고려한 valid 시작
        end_dt = df_sub['BAS_DT'].max()
        test_start_dt = shift_dt(end_dt, -(seq_length + 30))  # 마지막 데이터 기준 test 구간

        if tvt == 1:  # train
            print(f"train: {trk_start_dt.date()} ~ {train_end_dt.date()} 추출 중")
            df_sub = df_sub.loc[df_sub['BAS_DT'] <= train_end_dt, :].copy()

        elif tvt == 2:  # valid
            print(f"validation: {valid_start_dt.date()} ~ {dt_cut.date()} 추출 중")
            df_sub = df_sub.loc[(df_sub['BAS_DT'] >= valid_start_dt) & (df_sub['BAS_DT'] <= dt_cut), :].copy()

        else:  # test
            print(f"test: {test_start_dt.date()} ~ {end_dt.date()} 추출 중")
            df_sub = df_sub.loc[df_sub['BAS_DT'] >= test_start_dt, :].copy()

        x_past_list, x_known_list, x_static_list, y_list = [], [], [], []
        mean_val = scaler.mean_[ynum]
        std_val = scaler.scale_[ynum]
        for grp, group in df_sub.groupby(grp_cd, sort=False):
            if len(group) <= seq_length: continue

            group_past = group[past_vars].values
            group_known = group[known_vars].values.reshape(-1, 1)
            group_static = group[static_vars].iloc[0]

            for i in range(len(group) - seq_length):
                x_past_list.append(group_past[i:i+seq_length])
                x_known_list.append(group_known[i:i+seq_length])
                x_static_list.append(group_static)
                # t+1 시점의 target 값
                # y_list.append(group_past[i+seq_length, ynum])

                curr_val_scaled = group_past[i + seq_length - 1, ynum]
                next_val_scaled = group_past[i + seq_length, ynum]
                curr_val = (curr_val_scaled * std_val) + mean_val
                next_val = (next_val_scaled * std_val) + mean_val

                rate = (next_val / (curr_val + 1e-9)) - 1
                y_list.append(rate)

        # Array 변환
        x_past = np.array(x_past_list, np.float32)
        x_known = np.array(x_known_list, np.float32)
        x_static = np.array(x_static_list, np.int64)
        y = np.array(y_list, np.float32).reshape(-1, 1)

        return x_past, x_known, x_static, y

    # TVT 데이터 생성
    x_past_1, x_known_1, x_static_1, y_1 = make_sequence(df, 1)
    x_past_2, x_known_2, x_static_2, y_2 = make_sequence(df, 2)
    x_past_3, x_known_3, x_static_3, y_3 = make_sequence(df, 3)

    return ([x_past_1, x_past_2, x_past_3], 
            [x_known_1, x_known_2, x_known_3], 
            [x_static_1, x_static_2, x_static_3], 
            [y_1, y_2, y_3], 
            scaler, le)