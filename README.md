# GCY6902

1. File tree  
├── preprocess.py        # 시계열 데이터 전처리 (Scaling, Sliding Window 시퀀스 생성)  
├── tft_model.py           # TFT 모델 아키텍처 정의 (GRN, VSN, Attention 등)  
├── train_tft.py           # PyTorch Dataset 정의 및 학습/평가 루프 (Trainer)  
├── run.py                 # 전체 프로세스 통합 실행 함수 (fit_and_out)  
└── run_execute.ipynb       # 최종 실행용 Jupyter Notebook  

2. Flow
- 과거 binance 정보로 next timestamp 또는 long-term의 timestamp를 추정하는 구조
- TFT(Temporal Fusion Transformer) 아키텍처를 사용하여 변수 선택과 시계열 특징 추출을 동시에 수행

3.1 데이터 전처리 단계 (preprocess.py)
3.1.1 변수 구분: past_vars(과거 거래정보), known_vars(시간 인덱스), static_vars(symbol)로 분류
3.1.2 스케일링: StandardScaler를 사용하여 데이터를 표준화
3.1.3 시퀀스 생성: Sliding Window 기법을 사용하여 seq_length(기본 30일)만큼의 과거 데이터를 묶어 학습 샘플을 생성

3.2 TFT 모델 구조 (tft_model.py)
모델은 크게 4가지 핵심 블록으로 구성됩니다.
3.2.1 VSN (Variable Selection Network): 어떤 변수가 예측에 중요한지 가중치를 학습하여 선택적으로 정보를 반영합니다.
3.2.2 GRN (Gated Residual Network): 데이터의 비선형 특징을 효율적으로 추출하면서, 불필요한 층은 건너뛰는(Gating) 역할을 합니다.
3.2.3 LSTM Encoder: 시계열의 순차적인 흐름을 파악합니다.
3.2.4 Multi-head Attention: 과거 특정 시점의 데이터가 현재 예측에 얼마나 중요한지(장기 의존성) 파악합니다.

