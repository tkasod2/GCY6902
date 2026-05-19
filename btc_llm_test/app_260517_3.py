# conda activate gcy6902_llm
# cd btc_llm_test
# python -m streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(layout="wide", page_title="AI 트레이딩 시스템 (RAG + 대시보드)")

llm = Ollama(model="qwen2.5:3b")
embeddings = OllamaEmbeddings(model="qwen2.5:3b")

# ==========================================
# 2. 데이터 준비 (TFT 모델 3개 아웃풋 스펙 실시간 연동)
# ==========================================
@st.cache_data
def load_tft_inference_output():
    # 실행부 스크립트에서 자동 저장하도록 설정한 파일명 및 경로 정의
    report_file = "Backtest_Report_0405_to_0412.xlsx"
    trajectory_file = "Trajectory_Detail_0405_to_0412.xlsx"
    xai_file = "xai_weights_map.pkl"
    
    # 1단계: 마스터 요약 테이블(report_df) 동적 로드
    if os.path.exists(report_file):
        summary_df = pd.read_excel(report_file)
    else:
        # 파일이 없을 시 5개 코인 모델 체제에 맞춤화된 데모 데이터 생성
        summary_df = pd.DataFrame([
            {"Symbol": "BTCUSDT", "Decision": "🔴 강한 매수", "Score": 92, "Current_Price(4/5)": 65000.0, "Predicted_Min_Price": 64800.0, "Predicted_Max_Price": 71500.0, "Expected_Return_Pct": 10.3, "Best_Buy_Timing": "T+1", "Best_Sell_Timing": "T+21", "Strategy_Signal": "BUY"},
            {"Symbol": "SOLUSDT", "Decision": "🔴 매수", "Score": 85, "Current_Price(4/5)": 140.0, "Predicted_Min_Price": 138.5, "Predicted_Max_Price": 151.2, "Expected_Return_Pct": 8.0, "Best_Buy_Timing": "T+3", "Best_Sell_Timing": "T+18", "Strategy_Signal": "BUY"},
            {"Symbol": "ETHUSDT", "Decision": "🟡 보합", "Score": 55, "Current_Price(4/5)": 3500.0, "Predicted_Min_Price": 3490.0, "Predicted_Max_Price": 3535.0, "Expected_Return_Pct": 1.2, "Best_Buy_Timing": "T+1", "Best_Sell_Timing": "T+5", "Strategy_Signal": "HOLD"},
            {"Symbol": "XRPUSDT", "Decision": "🟡 보합", "Score": 50, "Current_Price(4/5)": 0.55, "Predicted_Min_Price": 0.54, "Predicted_Max_Price": 0.56, "Expected_Return_Pct": 0.8, "Best_Buy_Timing": "T+2", "Best_Sell_Timing": "T+4", "Strategy_Signal": "HOLD"},
            {"Symbol": "DOGEUSDT", "Decision": "🔴 매수", "Score": 78, "Current_Price(4/5)": 0.14, "Predicted_Min_Price": 0.135, "Predicted_Max_Price": 0.148, "Expected_Return_Pct": 5.4, "Best_Buy_Timing": "T+1", "Best_Sell_Timing": "T+12", "Strategy_Signal": "BUY"},
        ])
        
    # 2단계: (1)Symbol과 (3)타임스탬프별 가격 데이터셋(trajectory_df) 동적 로드
    if os.path.exists(trajectory_file):
        trajectory_df = pd.read_excel(trajectory_file)
    else:
        # 백업용 데모 궤적선 세팅 (21개 스탬프, 소문자 '8h' 적용)
        dates = pd.date_range(start="2026-04-05 08:00:00", periods=21, freq="8h")
        mock_records = []
        for _, row in summary_df.iterrows():
            sym = row['Symbol']
            base = row['Current_Price(4/5)']
            # 트렌드 생성
            pred_prices = np.linspace(base, base * (1 + row['Expected_Return_Pct']/100), 21)
            act_prices = pred_prices + np.random.normal(0, base * 0.01, 21)
            for idx, (p, a) in enumerate(zip(pred_prices, act_prices)):
                mock_records.append({
                    'Symbol': sym,
                    'Step': f"T+{idx+1}",
                    'BAS_DT': dates[idx].strftime('%Y%m%d%H'),
                    'Predicted_Price': p,
                    'Actual_Price': a,
                    'Pct_Change': ((p - base) / base) * 100
                })
        trajectory_df = pd.DataFrame(mock_records)
        
    # 3단계: (2) Symbol별 실전 XAI 데이터맵 로드
    if os.path.exists(xai_file):
        with open(xai_file, 'rb') as f:
            xai_dict = pickle.load(f)
    else:
        # 백업용 딕셔너리 구조 매핑
        xai_dict = {}
        for sym in summary_df['Symbol'].unique():
            xai_dict[sym] = pd.DataFrame({
                "Feature": ["Close", "fundingRate", "Volume", "High", "Low"],
                "Importance": [0.42, 0.23, 0.18, 0.10, 0.07]
            })
            
    return summary_df, trajectory_df, xai_dict

summary_df, trajectory_df, xai_dict = load_tft_inference_output()

# 4단계: 대시보드 메인 상단 비교 차트(Top 3 또는 전종목 누적 변동률) 생성
# trajectory_df의 Pct_Change 피처를 피벗하여 날짜 인덱스 기반으로 정렬합니다.
if 'Pct_Change' not in trajectory_df.columns:
    # 엑셀 로드본에 없을 경우 실시간 연산 보정
    trajectory_df['Pct_Change'] = trajectory_df.groupby('Symbol').apply(
        lambda x: ((x['Predicted_Price'] - x['Predicted_Price'].iloc[0]) / x['Predicted_Price'].iloc[0]) * 100
    ).reset_index(level=0, drop=True)

compare_df = trajectory_df.pivot(index='BAS_DT', columns='Symbol', values='Pct_Change')
compare_df.index = pd.to_datetime(compare_df.index, format='%Y%m%d%H')
compare_df.index.name = "Date"


# news_texts = [
#     "2025년 1월 20일 뉴스: 비트코인이 전 세계적인 기관 자금 유입에 힘입어 22K를 돌파했습니다. 특히 미국 월스트리트의 주요 펀드들이 암호화폐 비중을 확대하고 있습니다.",
#     "2025년 1월 21일 뉴스: 나스닥 지수가 3일 연속 상승하며 기술주와 동조화 현상을 보이는 암호화폐 시장도 전반적인 상승 랠리를 시작할 가능성이 높습니다.",
#     "2025년 1월 22일 뉴스: 미국 연방준비제도(Fed)가 금리를 동결할 것이라는 기대감이 커지면서, 크립토 시장의 펀딩비가 안정화되고 매수 심리가 회복되고 있습니다."
# ]
# news_links = [
#     {"title": "비트코인, 기관 자금 유입에 22K 돌파... 상승 랠리 시작되나?", "url": "https://www.coindesk.com/"},
#     {"title": "나스닥 훈풍에 암호화폐 시장 동반 상승세", "url": "https://kr.investing.com/"},
#     {"title": "미 연준 금리 동결 유력, 크립토 펀딩비 하락 안정화", "url": "https://cointelegraph.com/"}
# ]

# @st.cache_resource
# def create_vector_db():
#     docs = [Document(page_content=text) for text in news_texts]
#     vectorstore = FAISS.from_documents(docs, embeddings)
#     return vectorstore
# 
# vectorstore = create_vector_db()

prompt_template = PromptTemplate(
    input_variables=["portfolio_data", "context", "question"],
    template="""너는 데이터를 바탕으로 암호화폐 포트폴리오를 관리하는 전문 트레이딩 에이전트야.
    
    [현재 AI 모델 예측]
    {portfolio_data}
    
    사용자의 질문: {question}
    
    위 'AI 모델 예측' 수치들을 종합적으로 분석해서 답변해줘. 특히 기대수익률(Expected_Return_Pct)과 매매 타이밍 정보를 매핑해서 설명해줘.
    """
)

st.title("🤖 AI 암호화폐 트레이딩 시스템 (대시보드 + RAG)")

tab1, tab2 = st.tabs(["📊 요약 대시보드", "💬 대화형 에이전트"])

# ==========================================
# 탭 1: 요약 대시보드
# ==========================================
with tab1:
    st.subheader("💡 AI 모델 종합 종목 추천 및 전략 마스터 요약 리포트")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("**📈 자산별 주간 기대 변동률(%) 추정 궤적 비교**")
    st.info("기준일 가격을 0%로 설정하여, 모델이 예측한 각 종목의 상대적인 주간 자산 모멘텀 궤적을 비교합니다.")
    st.line_chart(compare_df, use_container_width=True)

    st.divider()

    # (1) 주입받은 개별 심볼 선택 드롭다운 박스
    st.subheader("🔍 개별 심볼 상세 분석 및 판단 근거")
    selected_symbol = st.selectbox("분석할 심볼을 선택하세요:", summary_df["Symbol"].tolist())
    
    # (1) + (3) 결합: 선택된 심볼의 타임스탬프별 가격 궤적(예상가격 vs 실제가격) 필터링 및 인덱스화
    sym_trajectory = trajectory_df[trajectory_df['Symbol'] == selected_symbol].copy()
    sym_trajectory['Date'] = pd.to_datetime(sym_trajectory['BAS_DT'], format='%Y%m%d%H')
    trend_df = sym_trajectory.set_index('Date')[['Predicted_Price', 'Actual_Price']]
    
    # (2) XAI 산출 데이터 연동: 딕셔너리에서 데이터프레임을 꺼내 정렬 및 인덱스 처리
    sym_xai = xai_dict[selected_symbol]
    if 'Feature' in sym_xai.columns:
        # 실전 인퍼런스 아웃풋 구조 대응 정형화
        reasoning_df = sym_xai.set_index('Feature')
    else:
        reasoning_df = sym_xai

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**1. {selected_symbol} 타임스탬프별 주간 가격 추정 궤적 및 정답 대조 ($)**")
        # 선형 차트에 TFT 모델의 예상 가격과 실전 정답 마켓 가격이 동시에 오버레이되어 드롭다운 변경 시 동적 실시간 변환
        st.line_chart(trend_df, use_container_width=True)
    with col2:
        st.markdown(f"**2. {selected_symbol} 판단 근거 (TFT Feature Importance)**")
        # 연산 완료된 실전 VSN 중요도가 막대그래프로 가변 매핑
        st.bar_chart(reasoning_df, use_container_width=True)
        
    st.divider()
    
    # st.markdown("📰 **실시간 주요 외부 뉴스 및 시황 (RAG 참조 데이터)**")
    # for news in news_links:
    #     st.markdown(f"- [{news['title']}]({news['url']})")

# ==========================================
# 탭 2: 대화형 에이전트
# ==========================================
with tab2:
    st.subheader("🗣️ 투자 전략 상담")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("질문 (예: 투자 시그널이 BUY인 자산들의 기대수익률과 청산 타이밍을 브리핑해줘)"):
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        context_str = "실시간 뉴스 연동 비활성화"
        
        # 동적 로드된 summary_df를 기반으로 가변 포트폴리오 프롬프트 생성
        portfolio_lines = []
        for _, row in summary_df.iterrows():
            signal = row.get('Strategy_Signal', 'HOLD')
            ret = row.get('Expected_Return_Pct', 0.0)
            b_time = row.get('Best_Buy_Timing', 'T+1')
            s_time = row.get('Best_Sell_Timing', 'T+21')
            portfolio_lines.append(f"- {row['Symbol']}: {signal} 시그널 (기대수익률 {ret:.2f}%, 매수지점:{b_time}, 청산지점:{s_time})")
        portfolio_str = "\n".join(portfolio_lines)

        formatted_prompt = prompt_template.format(
            portfolio_data=portfolio_str, 
            context=context_str, 
            question=user_question
        )

        with st.chat_message("assistant"):
            with st.spinner("데이터를 분석 중입니다..."):
                try:
                    response = llm.invoke(formatted_prompt)
                except Exception as e:
                    response = f"Ollama 커널 통신 중 예외가 발생했습니다: {e}"
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})