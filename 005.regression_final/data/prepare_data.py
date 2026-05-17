import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import gc


def prepare_data(df_in: pd.DataFrame,
                 target,
                 var_list,
                 trk_start_dt='2024030100',  # 시작 연월일시: YYYYMMDDHH
                 seq_length=21,              # 21개 시점, 8시간단위이므로 7일
                 dt_cut='2026030100',        # train/valid 분할 기준 연월일시: YYYYMMDDHH
                 grp_cd='Symbol',
                 tgt_gap=1):
    """
    df_in : Binance 데이터
    target : 예측 대상 컬럼명
    var_list : 입력 변수 리스트
    trk_start_dt : 관찰 시작 연월일시, YYYYMMDDHH
    seq_length : 몇 개 시점으로 예측할 것인지
                 시간 단위 데이터면 21 = 21시간
    dt_cut : train/valid 분할 기준 연월일시, YYYYMMDDHH
    grp_cd : 그룹 코드 컬럼명
    tgt_gap : 예측 시차
        ex) tgt_gap = 0 : t까지의 데이터로 t+1 시점 예측
            tgt_gap = 3 : t까지의 데이터로 t+4 시점 예측
    """

    past_vars = var_list

    # 문자열 기준 비교를 위해 BAS_DT를 10자리 문자열로 통일
    df = df_in.copy()
    df['BAS_DT'] = df['BAS_DT'].astype(str).str.zfill(10)

    trk_start_dt = str(trk_start_dt).zfill(10)
    dt_cut = str(dt_cut).zfill(10)

    df = df.loc[df['BAS_DT'] >= trk_start_dt, :].copy()
    df = df.sort_values(by=[grp_cd, 'BAS_DT']).reset_index(drop=True)

    # 전체 입력변수 스케일링
    scaler = StandardScaler()
    scaler.fit(df.loc[df['BAS_DT'] <= dt_cut, past_vars])

    raw_target_series = df[target].copy()

    df[past_vars] = scaler.transform(df[past_vars])

    # target 위치 확인
    dct_target = {col: idx for idx, col in enumerate(past_vars)}
    ynum = dct_target[target]

    # 그룹별 target scaler 생성
    target_scalers = {}

    for g_name, group_idx in df.groupby(grp_cd).groups.items():
        g_scaler = StandardScaler()

        train_mask = df.loc[group_idx, 'BAS_DT'] <= dt_cut
        train_target_raw = raw_target_series.loc[group_idx][train_mask].values.reshape(-1, 1)

        if len(train_target_raw) > 0:
            g_scaler.fit(train_target_raw)
        else:
            g_scaler.mean_ = np.array([scaler.mean_[ynum]])
            g_scaler.scale_ = np.array([scaler.scale_[ynum]])

        target_scalers[g_name] = g_scaler

        raw_group_all = raw_target_series.loc[group_idx].values.reshape(-1, 1)
        df.loc[group_idx, target] = g_scaler.transform(raw_group_all).flatten()

    df[past_vars] = df[past_vars].fillna(0)

    # Known 공변량: 시간 index
    df['tmp_dt'] = pd.to_datetime(df['BAS_DT'], format='%Y%m%d%H')
    base_dt = pd.to_datetime(trk_start_dt, format='%Y%m%d%H')

    df['dt_idx'] = ((df['tmp_dt'] - base_dt).dt.total_seconds() // (3600*8)).astype(int)

    del df['tmp_dt']
    # known_vars = ['dt_idx','day_of_week','hour']
    known_vars = ['dt_idx']
    # Static 공변량: 그룹 코드
    le = LabelEncoder()
    le.fit(df[grp_cd])

    df['grp_cd_idx'] = le.transform(df[grp_cd])
    static_vars = ['grp_cd_idx']

    # 최종 컬럼 정리
    lst_final_cols = ['BAS_DT', grp_cd, target] + var_list + known_vars + static_vars
    lst_final_cols = list(dict.fromkeys(lst_final_cols))

    df = df[lst_final_cols].copy()
    gc.collect()

    print(f"====target : {ynum + 1} {target}=====")

    def shift_dt(dt_val, hours):
        """
        YYYYMMDDHH를 받아서 hours만큼 이동 후 YYYYMMDDHH 문자열 반환
        """
        dt = pd.to_datetime(str(dt_val).zfill(10), format='%Y%m%d%H')
        shifted = dt + pd.Timedelta(hours=hours)
        return shifted.strftime('%Y%m%d%H')

    def make_sequence(df_in, tvt=1):
        df = df_in.copy()

        # 시간 단위 기준
        test_valid_period = 24 * 28 + 24 * 31

        train_end_dt = shift_dt(dt_cut, -test_valid_period)

        valid_start_dt = shift_dt(train_end_dt, -(seq_length + tgt_gap) * 8)

        end_dt = df['BAS_DT'].max()
        test_start_raw = '2026030100'
        test_start_dt = shift_dt(test_start_raw, -(seq_length + tgt_gap) * 8)

        if tvt == 1:
            df = df.loc[df['BAS_DT'] <= train_end_dt, :].copy()
            print(f"train: {trk_start_dt}~{train_end_dt} 자료 추출 중 : {len(df)}")

        elif tvt == 2:
            df = df.loc[
                (df['BAS_DT'] >= valid_start_dt) &
                (df['BAS_DT'] <= dt_cut),
                :
            ].copy()
            print(f"validation set : {valid_start_dt}~{dt_cut} 자료 추출 중 : {len(df)}")

        else:
            df = df.loc[df['BAS_DT'] >= test_start_dt, :].copy()
            print(f"test set : {test_start_dt}~{end_dt} 자료 추출 중 : {len(df)}")

        x_past_list, x_known_list, x_static_list, y_list = [], [], [], []

        for g_name, group in df.groupby(grp_cd, sort=False):
            if len(group) < seq_length + tgt_gap + 1:
                continue

            mean_y = target_scalers[g_name].mean_[0]
            std_y = target_scalers[g_name].scale_[0]

            group_past = group[past_vars].values
            group_known = group[known_vars].values.reshape(-1, 1)
            group_static = group[static_vars].iloc[0]

            for i in range(len(group) - seq_length - tgt_gap):
                x_past_list.append(group_past[i:i + seq_length])
                x_known_list.append(group_known[i:i + seq_length])
                x_static_list.append(group_static)

                y_val = group_past[i + seq_length + tgt_gap, ynum]

                y_base_scaled = group_past[i + seq_length - 1, ynum]

                raw_base = y_base_scaled * std_y + mean_y
                raw_target = y_val * std_y + mean_y

                diff = (raw_target - raw_base) / (raw_base + 1e-9)

                if diff > 0.02:
                    y_cls = 1
                elif diff < -0.02:
                    y_cls = 2
                else:
                    y_cls = 0

                y_list.append([y_val, y_cls])

        x_past = np.array(x_past_list, np.float32)
        x_known = np.array(x_known_list, int)
        x_static = np.array(x_static_list, int)
        y = np.array(y_list, np.float32)

        return x_past, x_known, x_static, y

    x_past_1, x_known_1, x_static_1, y_1 = make_sequence(df, 1)
    x_past_2, x_known_2, x_static_2, y_2 = make_sequence(df, 2)
    x_past_3, x_known_3, x_static_3, y_3 = make_sequence(df, 3)

    x_past = [x_past_1, x_past_2, x_past_3]
    x_known = [x_known_1, x_known_2, x_known_3]
    x_static = [x_static_1, x_static_2, x_static_3]
    y = [y_1, y_2, y_3]

    return x_past, x_known, x_static, y, scaler, le, target_scalers