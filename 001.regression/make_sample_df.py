import numpy as np
import pandas as pd

def make_sample_df(
    n_industries=80,
    strt_trk_ym=201101,
    end_ym=202112,
    n_features=65,
    seed=42
):
    np.random.seed(seed)

    yms = np.arange(strt_trk_ym, end_ym + 1)
    valid_idx = np.divmod(yms,100)[1]<=12
    yms = yms[valid_idx]
    rows = []

    # 산업코드 예시
    industry_codes = [f"{i:03d}" for i in range(1, n_industries + 1)]

    for ind_idx, code in enumerate(industry_codes):
        # 산업별 고정효과
        industry_effect = np.random.normal(0, 0.5)

        # 각 산업별 base level
        base = np.random.normal(0, 1, n_features)

        for ym in yms:
            year_trend = (ym - strt_trk_ym) * 0.05
            noise = np.random.normal(0, 0.3, n_features)

            # 재무변수 F001~F065 생성
            vals = base + industry_effect + year_trend + noise

            row = {
                "BAS_YM": ym,
                "SIC_CD": code,
            }

            for j in range(n_features):
                row[f"F{j+1:03d}"] = vals[j]

            rows.append(row)

    df = pd.DataFrame(rows)

    # 약간의 결측 추가(현재 코드에서 fillna(0) 처리하므로 테스트 가능)
    # feat_cols = [f"F{i:03d}" for i in range(1, n_features + 1)]
    # mask = np.random.rand(len(df), len(feat_cols)) < 0.02
    # df.loc[:, feat_cols] = df.loc[:, feat_cols].mask(mask)

    return df
