import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import gc

def prepare_data(df_in: pd.DataFrame,
                 target,
                 var_list,
                 trk_start_ym=201101,
                 seq_length=60,  # 60개월
                 ym_cut=201912,
                 grp_cd = 'SIC_CD',
                 output_mode = 'multiclass',
                 thr=0.10):
    
    """
    df_in : 산업재무 데이터
    target : 추정하고자하는 산업재무 항목
    trk_start_ym : 보유정보 최초의 관찰 시작 연월
    seq_length : n개월 데이터로 추정할것인지에 대한 params
    ym_cut : train/valid 데이터의 max 연월
    
    """

    # 1. Data 세팅
    # (1) 일반적인 공변량 : 산업재무비율정보
    # 
    past_vars = var_list
    df = df_in.loc[df_in['BAS_YM'] >= trk_start_ym, :].copy()
    df = df.sort_values(by=[grp_cd, 'BAS_YM']).reset_index(drop=True)

    scaler = StandardScaler()

    scaler.fit(df.loc[df['BAS_YM'] <= ym_cut, past_vars])  # train data로 fitting
    df[past_vars] = scaler.transform(df[past_vars])  # 전체 데이터에 대해서 scaling 진행
    
    df[past_vars] = df[past_vars].fillna(0)  # 결측치 보간은 우선 평균값 대치
    df = df.sort_values(by=['SIC_CD', 'BAS_YM'])

    # (2) Known 공변량 : 연도 index
    df['ym_idx'] = df['BAS_YM'] - trk_start_ym
    a_, b_ = np.divmod(df['ym_idx'], 100)    
    df['ym_idx'] = 12*a_+b_
    df['ym_idx'] = df['ym_idx'].astype(int)
    known_vars = ['ym_idx']

    # (3) Static 공변량 : 산업분류코드
    le = LabelEncoder()
    df['sic_cd_idx'] = le.fit_transform(df[grp_cd])
    static_vars = ['sic_cd_idx']

    # (4) target (부도율)
    lst_final_cols = ['BAS_YM', grp_cd, target] + var_list + known_vars + static_vars
    lst_final_cols = list(dict.fromkeys(lst_final_cols))
    df = df[lst_final_cols].copy()
    gc.collect()

    #target 순서 확인
    dct_target = {col: idx for idx, col in enumerate(past_vars)}
    ynum = dct_target[target]
    print(f"=====target : {ynum+1} {target}=====")

    # (5) 시퀀스 생성
    # 260404 수정내용
    def shift_ym(ym, months):
        """정수형 YYYYMM을 받아서 months만큼 이동시킨 후 다시 정수형 YYYYMM 반환"""
        dt = pd.to_datetime(str(ym), format='%Y%m')
        shifted = dt + pd.DateOffset(months=months)
        return int(shifted.strftime('%Y%m'))
    
    def classify_movement_raw(current_raw, base_raw, thr=0.10):
        diff = (current_raw - base_raw) / (base_raw + 1e-9)
        if diff > thr:
            return 2   # 상승
        elif diff < -thr:
            return 0   # 하락
        else:
            return 1   # 유지
    
    def make_sequence(df_in, tvt=1):

        df = df_in.copy()
        train_end_ym = shift_ym(ym_cut, -12)        # ym cut 기준으로 1년 전까지를 train set
        valid_start_ym = shift_ym(shift_ym(ym_cut,1), -72)      # ym cut 기준으로 5년 전부터 valid set (시퀀스 길이 확보용)
        end_ym = df['BAS_YM'].max()
        test_start_ym = shift_ym(shift_ym(end_ym,1), -72)  # max BAS_YM 기준으로 1년 기간 test set (시퀀스 길이 확보용)

        if tvt == 1:  # train
            print(f"train: {trk_start_ym}~{train_end_ym} 자료 추출 중")
            df = df.loc[df['BAS_YM'] <= train_end_ym, :].copy()


        elif tvt == 2:  # valid
            print(f"validation set : {valid_start_ym}~{ym_cut} 자료 추출 중")
            df = df.loc[(df['BAS_YM'] >= valid_start_ym) &(df['BAS_YM'] <= ym_cut),:].copy()


        else:  # test
            print(f"test set : {test_start_ym}~{end_ym} 자료 추출 중")
            df = df.loc[df['BAS_YM'] >= test_start_ym, :].copy()

        # df_raw = df.copy()
        # df_raw[past_vars] = scaler.inverse_transform(df_raw[past_vars])

        x_past_list, x_known_list, x_static_list, y_list, y_class = [], [], [], [], []

        for sic_cd, group in df.groupby(grp_cd, sort=False):
            if len(group) < seq_length: continue

            group_past = group[past_vars].values
            group_known = group[known_vars].values.reshape(-1, 1)
            group_static = group[static_vars].iloc[0]
            group_raw = scaler.inverse_transform(group[past_vars].values)
            

            for i in range(len(group) - seq_length):

                x_past_list.append(group_past[i:i+seq_length])
                x_known_list.append(group_known[i:i+seq_length])
                x_static_list.append(group_static)

                y_list.append(group_past[i+seq_length, ynum])
                current_raw = group_raw[i+seq_length, ynum]
                base_raw = group_raw[i+seq_length-1, ynum]
                y_cls = classify_movement_raw(current_raw, base_raw, thr=thr)
                y_class.append(y_cls)

        x_past = np.array(x_past_list, np.float32)
        x_known = np.array(x_known_list, int)
        x_static = np.array(x_static_list, int)
        if output_mode == 'regression':
            y = np.array(y_list, np.float32).reshape(-1, 1)
        else :
            y = np.array(y_class, dtype=np.int64)
            print(f"Class distribution: {np.bincount(y)}")
        return x_past, x_known, x_static, y

    x_past_1, x_known_1, x_static_1, y_1 = make_sequence(df, 1)
    x_past_2, x_known_2, x_static_2, y_2 = make_sequence(df, 2)
    x_past_3, x_known_3, x_static_3, y_3 = make_sequence(df, 3)

    x_past = [x_past_1, x_past_2, x_past_3]
    x_known = [x_known_1, x_known_2, x_known_3]
    x_static = [x_static_1, x_static_2, x_static_3]
    y = [y_1, y_2, y_3]
    return x_past, x_known, x_static, y, scaler, le