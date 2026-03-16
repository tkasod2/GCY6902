import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data(df_in:pd.DataFrame,
                 target='Close',
                 seq_length=30,
                 test_date ='2025-03-01'):

    """
    df_in : e데이터
    target : 예측 가격(일종가, 4시간봉 등등)
    seq_length : 하나의 SEQ를 몇줄로 할것인지
    tvt_date : train&valid // test를 나누는 경계 시점
    """

    # 1. Data 세팅
    # (1) 일반적인 공변량 : 산업재무비율정보
    past_vars = ['Open', 'High', 'Low', 'Close', 'Volume', 
                 'Quote asset volume', 'Number of trades', 
                 'Taker buy base asset volume', 'Taker buy quote asset volume']

    df = df_in.reset_index().copy() 
    time_col = 'Close time'

    scaler = StandardScaler()

    scaler.fit(df.loc[df[time_col] < '2024-12-01', past_vars])  # train data로 standard scaler fitting
    df[past_vars] = scaler.transform(df[past_vars])  # 전체 데이터에 대해서 scaling 진행
    df[past_vars] = df[past_vars].fillna(0)  # 결측치 보간은 우선 0으로 처리

    # (2) Known 공변량 : 연도-최초추적관찰연도
    df['hour'] = df[time_col].dt.hour
    df['dayofweek'] = df[time_col].dt.dayofweek
    known_vars = ['hour', 'dayofweek']

    # (3) Static 공변량 : Symbol
    le = LabelEncoder()
    df['symbol_idx'] = le.fit_transform(df['Symbol'])
    static_vars = ['symbol_idx']

    # (4) target(재무비율)
    target_idx = past_vars.index(target)

    # (5) 시퀀스 생성(sliding window)

    def make_sequence(df_in):

        df = df_in.copy()
        x_past_list, x_known_list, x_static_list, y_list = [], [], [], []

        for symbol, group in df.groupby(['Symbol']):
            if len(group) < seq_length: continue
            
            group_past = group[past_vars].values
            group_known = group[known_vars].values # (L, 2)
            group_static = group['symbol_idx'].iloc[0]

            for i in range(len(group) - seq_length):
                x_past_list.append(group_past[i:i+seq_length])
                x_known_list.append(group_known[i:i+seq_length])
                x_static_list.append(group_static)
                y_list.append(group_past[i+seq_length, target_idx])

        return (np.array(x_past_list, np.float32), 
                np.array(x_known_list, np.float32), 
                np.array(x_static_list, np.float32).reshape(-1, 1), 
                np.array(y_list, np.float32).reshape(-1, 1))
    
    tv_df = df[df[time_col] < test_date].copy()
    train_df, valid_df = train_test_split(tv_df,test_size=0.2,random_state=42)
    test_df = df[df[time_col] >= test_date].copy()

    x_past_1, x_known_1, x_static_1, y_1 = make_sequence(train_df)
    x_past_2, x_known_2, x_static_2, y_2 = make_sequence(valid_df)
    x_past_3, x_known_3, x_static_3, y_3 = make_sequence(test_df)

    return [x_past_1, x_past_2, x_past_3], [x_known_1, x_known_2, x_known_3], \
           [x_static_1, x_static_2, x_static_3], [y_1, y_2, y_3], scaler, le