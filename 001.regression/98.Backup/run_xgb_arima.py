# -*- coding: utf-8 -*-
import os
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

from prepare_data import prepare_data
from make_sample_df import make_sample_df

today_ = datetime.now().strftime("%y%m%d")


def classify_movement(current, base, thr=0.05):
    diff = (current - base) / (base + 1e-9)
    if diff > thr:
        return "상승"
    elif diff < -thr:
        return "하락"
    else:
        return "유지"


def inverse_transform_target(arr, scaler, target):
    """
    target 변수 하나만 원복
    arr: np.ndarray shape (N,) 또는 (N,1)
    """
    arr = np.asarray(arr).reshape(-1)
    df_scaler = pd.DataFrame({
        "VAR": scaler.feature_names_in_,
        "mean": scaler.mean_,
        "std": scaler.scale_
    })
    mean_val = df_scaler.loc[df_scaler["VAR"] == target, "mean"].item()
    std_val = df_scaler.loc[df_scaler["VAR"] == target, "std"].item()
    return arr * std_val + mean_val


def make_xgb_features(x_past, x_known=None, x_static=None):
    """
    XGBoost 입력용 평탄화
    x_past  : (N, L, V)
    x_known : (N, L, K) 또는 None
    x_static: (N, S) 또는 None
    """
    n = x_past.shape[0]
    feat_list = [x_past.reshape(n, -1)]

    if x_known is not None and len(x_known) > 0:
        feat_list.append(x_known.reshape(n, -1))

    if x_static is not None and len(x_static) > 0:
        feat_list.append(x_static.reshape(n, -1))

    return np.concatenate(feat_list, axis=1)


def fit_xgboost_and_arima(
    df,
    target,
    outdir,
    var_list,
    grp_cd="SIC_CD",
    threshold=0.05,
    trk_start_ym=201101,
    seq_length=60,
    ym_cut=201912,
    xgb_params=None,
    arima_order=(1, 1, 1)
):
    os.makedirs(outdir, exist_ok=True)

    # 1) 기존 전처리 재사용
    x_past, x_known, x_static, y, scaler, le = prepare_data(
        df_in=df,
        target=target,
        var_list=var_list,
        trk_start_ym=trk_start_ym,
        seq_length=seq_length,
        ym_cut=ym_cut,
        grp_cd=grp_cd
    )

    x_train, x_valid, x_test = x_past[0], x_past[1], x_past[2]
    k_train, k_valid, k_test = x_known[0], x_known[1], x_known[2]
    s_train, s_valid, s_test = x_static[0], x_static[1], x_static[2]
    y_train, y_valid, y_test = y[0].reshape(-1), y[1].reshape(-1), y[2].reshape(-1)

    # -----------------------------
    # 2) XGBoost
    # -----------------------------
    X_train = make_xgb_features(x_train, k_train, s_train)
    X_valid = make_xgb_features(x_valid, k_valid, s_valid)
    X_test = make_xgb_features(x_test, k_test, s_test)

    if xgb_params is None:
        xgb_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42
        }

    model_xgb = XGBRegressor(**xgb_params)
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    pred_xgb = model_xgb.predict(X_test)

    # -----------------------------
    # 3) ARIMA
    # -----------------------------
    # ARIMA는 외생변수 없이 target 단변량으로만 비교
    # 현재 test 샘플은 "산업별 rolling window" 구조라서,
    # 각 test sample의 마지막 seq 구간(target 시계열)로부터 1-step forecast 수행
    target_idx = var_list.index(target)
    pred_arima = []

    for i in range(x_test.shape[0]):
        hist = x_test[i, :, target_idx]  # scaled target history, shape (L,)
        try:
            fit = ARIMA(hist, order=arima_order).fit()
            fcst = fit.forecast(steps=1)
            pred_arima.append(float(fcst[0]))
        except Exception:
            # 실패 시 마지막 값 naive fallback
            pred_arima.append(float(hist[-1]))

    pred_arima = np.array(pred_arima)

    # -----------------------------
    # 4) 원 단위(스케일 복원)
    # -----------------------------
    actual_raw = inverse_transform_target(y_test, scaler, target)
    xgb_raw = inverse_transform_target(pred_xgb, scaler, target)
    arima_raw = inverse_transform_target(pred_arima, scaler, target)

    # t-1 값 복원
    y_t_minus_1_raw = inverse_transform_target(x_test[:, -1, target_idx], scaler, target)

    # grp_cd 복원
    grp_vals = le.inverse_transform(s_test.reshape(-1).astype(int))

    # TIME_IDX
    time_idx = k_test[:, -1, 0] + 1

    res_df = pd.DataFrame({
        "TIME_IDX": time_idx,
        grp_cd: grp_vals,
        "Actual": actual_raw,
        "XGB_Prediction": xgb_raw,
        "ARIMA_Prediction": arima_raw,
        "Actual_t_minus_1": y_t_minus_1_raw
    })

    # 방향성
    res_df["Actual_Move"] = res_df.apply(
        lambda x: classify_movement(x["Actual"], x["Actual_t_minus_1"], threshold), axis=1
    )
    res_df["XGB_Move"] = res_df.apply(
        lambda x: classify_movement(x["XGB_Prediction"], x["Actual_t_minus_1"], threshold), axis=1
    )
    res_df["ARIMA_Move"] = res_df.apply(
        lambda x: classify_movement(x["ARIMA_Prediction"], x["Actual_t_minus_1"], threshold), axis=1
    )

    # -----------------------------
    # 5) 성능평가
    # -----------------------------
    xgb_rmse = np.sqrt(mean_squared_error(res_df["Actual"], res_df["XGB_Prediction"]))
    xgb_mae = mean_absolute_error(res_df["Actual"], res_df["XGB_Prediction"])
    xgb_acc = accuracy_score(res_df["Actual_Move"], res_df["XGB_Move"])

    arima_rmse = np.sqrt(mean_squared_error(res_df["Actual"], res_df["ARIMA_Prediction"]))
    arima_mae = mean_absolute_error(res_df["Actual"], res_df["ARIMA_Prediction"])
    arima_acc = accuracy_score(res_df["Actual_Move"], res_df["ARIMA_Move"])

    df_pfmc = pd.DataFrame({
        "model": ["XGBoost", "ARIMA"],
        "rmse": [xgb_rmse, arima_rmse],
        "mae": [xgb_mae, arima_mae],
        "directional_accuracy": [xgb_acc, arima_acc]
    })

    # XGBoost 변수중요도
    feature_names = []
    L = x_train.shape[1]
    V = x_train.shape[2]

    for t in range(L):
        for v in var_list:
            feature_names.append(f"{v}_lag{L - t}")

    if k_train is not None and len(k_train) > 0:
        k_dim = k_train.shape[2]
        for t in range(L):
            for j in range(k_dim):
                feature_names.append(f"known_{j+1}_lag{L - t}")

    if s_train is not None and len(s_train) > 0:
        s_dim = s_train.shape[1]
        for j in range(s_dim):
            feature_names.append(f"static_{j+1}")

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model_xgb.feature_importances_
    }).sort_values("Importance", ascending=False)

    # 산업별 성능
    ind_stats = []
    for g, sub in res_df.groupby(grp_cd):
        ind_stats.append({
            grp_cd: g,
            "sample_count": len(sub),
            "xgb_rmse": np.sqrt(mean_squared_error(sub["Actual"], sub["XGB_Prediction"])),
            "xgb_mae": mean_absolute_error(sub["Actual"], sub["XGB_Prediction"]),
            "xgb_hit_rate": accuracy_score(sub["Actual_Move"], sub["XGB_Move"]),
            "arima_rmse": np.sqrt(mean_squared_error(sub["Actual"], sub["ARIMA_Prediction"])),
            "arima_mae": mean_absolute_error(sub["Actual"], sub["ARIMA_Prediction"]),
            "arima_hit_rate": accuracy_score(sub["Actual_Move"], sub["ARIMA_Move"])
        })
    df_ind_pfmc = pd.DataFrame(ind_stats)

    # 저장
    res_path = os.path.join(outdir, f"{target}_xgb_arima_result_{today_}.csv")
    pfmc_path = os.path.join(outdir, f"{target}_xgb_arima_performance_{today_}.csv")
    imp_path = os.path.join(outdir, f"{target}_xgb_feature_importance_{today_}.csv")
    ind_path = os.path.join(outdir, f"{target}_xgb_arima_by_industry_{today_}.csv")

    res_df.to_csv(res_path, index=False, encoding="utf-8-sig")
    df_pfmc.to_csv(pfmc_path, index=False, encoding="utf-8-sig")
    imp_df.to_csv(imp_path, index=False, encoding="utf-8-sig")
    df_ind_pfmc.to_csv(ind_path, index=False, encoding="utf-8-sig")

    print("===== 모델 성능 =====")
    print(df_pfmc)

    return {
        "xgb_model": model_xgb,
        "result_df": res_df,
        "performance_df": df_pfmc,
        "importance_df": imp_df,
        "industry_performance_df": df_ind_pfmc
    }


if __name__ == "__main__":
    # 예시 데이터
    df = make_sample_df(
        n_industries=30,
        strt_trk_ym=201101,
        end_ym=202112,
        n_features=65,
        seed=42
    )

    var_list = [f"F{i:03d}" for i in range(1, 66)]
    target = "F001"

    out = fit_xgboost_and_arima(
        df=df,
        target=target,
        outdir="./output_xgb_arima/",
        var_list=var_list,
        grp_cd="SIC_CD",
        threshold=0.05,
        trk_start_ym=201101,
        seq_length=60,
        ym_cut=201912,
        xgb_params={
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42
        },
        arima_order=(1, 1, 1)
    )