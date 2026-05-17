# conda activate gcy6902_llm
# python -m streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# =========================================================================
# 1. 화면 전체 레이아웃 설정
# =========================================================================
st.set_page_config(layout="wide", page_title="TFT AI Trading Dashboard")
st.title("🤖 TFT 주간 궤적 추정 & 뉴스 감성분석 트레이딩 시스템")

# =========================================================================
# 2. 로컬 LLM 및 임베딩 커널 설정 (Ollama - Qwen2.5:3b)
# =========================================================================
@st.cache_resource
def init_llm_pipeline():
    llm = Ollama(model="qwen2.5:3b")
    embeddings = OllamaEmbeddings(model="qwen2.5:3b")
    return llm, embeddings

try:
    llm, embeddings = init_llm_pipeline()
except Exception as e:
    st.error(f"⚠️ Ollama 커널 로드 실패 (로컬 백엔드 점검 필요): {e}")

# =========================================================================
# 3. 뉴스 감성분석 파일(CSV) 로드 파이프라인
# =========================================================================
@st.cache_data
def load_news_sentiment_data():
    # app.py와 같은 경로에 있으므로 파일명만 지정
    file_name = "news_sentiment_8h.csv"
    if os.path.exists(file_name):
        # 16만 건을 한 번에 읽지 않고, 20,000행씩 쪼개서 메모리 버퍼에서 순식간에 처리합니다.
        # 메모리 낭비를 줄이기 위해 필요한 핵심 열(usecols)만 명시하여 수집 속도를 극대화합니다.
        target_cols = ['coin', 'title', 'date', 'bucket_8h', 'source', 'url', 'pos', 'neg', 'neu', 'score']
        
        april_chunks = []
        try:
            # chunksize를 선언하면 파일 전체를 RAM에 적재하지 않고 이터레이터로 읽어옵니다.
            for chunk in pd.read_csv(file_name, chunksize=20000, usecols=target_cols, encoding='utf-8'):
                chunk = chunk.dropna(subset=['coin', 'title', 'date'])
                
                # 2026년 4월에 해당하는 데이터만 조각별로 필터링
                chunk['date'] = chunk['date'].astype(str)
                filtered_chunk = chunk[chunk['date'].str.startswith('2026-04')]
                
                if not filtered_chunk.empty:
                    april_chunks.append(filtered_chunk)
                    
            if april_chunks:
                df_april_news = pd.concat(april_chunks, ignore_index=True)
                return df_april_news
            else:
                # 2026년 4월 데이터가 통계상 아예 없을 경우를 대비해 맨 상위 조각만 안전 백업용 수령
                st.info("ℹ️ 현재 데이터셋 내에 '2026-04' 데이터가 발견되지 않아, 상위 200개 데이터셋을 임시 서빙합니다.")
                return pd.read_csv(file_name, nrows=200, usecols=target_cols)
                
        except Exception as e:
            st.warning(f"⚠️ 인코딩 또는 파싱 예외 발생으로 가볍게 재시도합니다: {e}")
            return pd.read_csv(file_name, nrows=200, usecols=target_cols, errors='ignore')
    else:
        st.warning(f"⚠️ 폴더 내에 '{file_name}' 파일이 없습니다. 기본 데모 텍스트로 구동합니다.")
        return pd.DataFrame()

df_news = load_news_sentiment_data()

# =========================================================================
# 4. TFT 모델 백테스트 결과 엑셀 파일 로드 파이프라인
# =========================================================================
@st.cache_data
def load_tft_inference_output():
    # app.py와 같은 경로에 있으므로 파일명만 지정
    report_file = "Backtest_Report_0405_to_0412.xlsx"
    
    if os.path.exists(report_file):
        summary_df = pd.read_excel(report_file)
    else:
        # 엑셀 파일이 없을 경우 화면이 꺼지는 것을 방지하는 데모용 데이터셋 자동 빌드
        summary_df = pd.DataFrame([
            {"Symbol": "BTCUSDT", "Current_Price(4/5)": 65000.0, "Predicted_Min_Price": 64800.0, "Predicted_Max_Price": 71500.0, "Expected_Return_Pct": 10.3, "Best_Buy_Timing": "T+1", "Best_Sell_Timing": "T+21", "Strategy_Signal": "BUY"},
            {"Symbol": "SOLUSDT", "Current_Price(4/5)": 140.0, "Predicted_Min_Price": 138.5, "Predicted_Max_Price": 151.2, "Expected_Return_Pct": 8.0, "Best_Buy_Timing": "T+3", "Best_Sell_Timing": "T+18", "Strategy_Signal": "BUY"},
            {"Symbol": "ETHUSDT", "Current_Price(4/5)": 3500.0, "Predicted_Min_Price": 3490.0, "Predicted_Max_Price": 3535.0, "Expected_Return_Pct": 1.0, "Best_Buy_Timing": "T+1", "Best_Sell_Timing": "T+5", "Strategy_Signal": "HOLD"},
        ])
        
    # 21개 스탬프 동안의 자산별 변동성 추정 궤적선 시뮬레이션 데이터 구축
    dates = pd.date_range(start="2026-04-05 08:00:00", periods=21, freq="8h")
    compare_dict = {}
    xai_dict = {}
    
    for _, row in summary_df.head(5).iterrows():
        sym = row['Symbol']
        # 기대 수익률을 기반으로 21개 스탬프 상의 라인 그래프 생성
        compare_dict[sym] = np.linspace(0, row['Expected_Return_Pct'], 21) + np.random.normal(0, 0.2, 21)
        # 중요 변수 가중치 매핑
        xai_dict[sym] = pd.DataFrame({"기여도": [0.42, 0.23, 0.18, 0.10, 0.07]}, index=["Close", "fundingRate", "Volume", "High", "Low"])
        
    compare_df = pd.DataFrame(compare_dict, index=dates)
    return summary_df, compare_df, xai_dict

summary_df, compare_df, xai_dict = load_tft_inference_output()

# =========================================================================
# 5. 뉴스 감성분석 데이터를 기반으로 RAG 전용 Vector DB 빌드
# =========================================================================
@st.cache_resource
def build_rag_vector_store(_df_news):
    if _df_news.empty:
        default_docs = [Document(page_content="비트코인 펀딩비가 안정세에 진입하며 선물 시장의 과열이 해소되었습니다.")]
        return FAISS.from_documents(default_docs, embeddings)
    
    docs = []
    for _, row in _df_news.iterrows():
        # LLM이 뉴스의 텍스트 맥락과 수치적 감성 점수를 함께 파악하도록 통합 컨텍스트 정형화
        context_text = (
            f"코인: {row['coin']} | 뉴스제목: {row['title']} | 출처: {row['source']} | "
            f"감성지표(긍정:{row['pos']:.2f}, 부정:{row['neg']:.2f}, 중립:{row['neu']:.2f}) | "
            f"종합감성점수: {row['score']}"
        )
        docs.append(Document(page_content=context_text, metadata={"url": row['url'], "title": row['title']}))
    
    return FAISS.from_documents(docs, embeddings)

vectorstore = build_rag_vector_store(df_news)

# 6. 대시보드 탭 레이아웃 분할
tab1, tab2 = st.tabs(["📊 요약 대시보드 (TFT + Sentiment)", "💬 AI 감성분석 RAG 에이전트"])

# ==========================================
# 탭 1: 요약 대시보드
# ==========================================
with tab1:
    st.subheader("💡 AI 모델 추천 주간 투자 전략 시그널")
    st.dataframe(
        summary_df[['Symbol', 'Current_Price(4/5)', 'Predicted_Min_Price', 'Predicted_Max_Price', 'Best_Buy_Timing', 'Best_Sell_Timing', 'Expected_Return_Pct', 'Strategy_Signal']].head(5),
        use_container_width=True, hide_index=True
    )
    
    st.markdown("**📈 자산별 주간 기대 변동률(%) 추정 궤적 비교**")
    st.line_chart(compare_df, use_container_width=True)
    
    st.divider()
    
    st.subheader("🔍 개별 자산 심층 판단 근거")
    selected_symbol = st.selectbox("심층 분석할 자산을 선택하세요:", summary_df["Symbol"].tolist())
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**1. [{selected_symbol}] 모델 추천 매매 타이밍**")
        sym_info = summary_df[summary_df['Symbol'] == selected_symbol].iloc[0]
        st.metric(label="최적 매수 진입 시점", value=sym_info['Best_Buy_Timing'])
        st.metric(label="최적 청산(매도) 시점", value=sym_info['Best_Sell_Timing'], delta=f"기대수익률: {sym_info['Expected_Return_Pct']:.2f}%")
    with col2:
        st.markdown(f"**2. [{selected_symbol}] 공변량 기여도 (Feature Importance)**")
        if selected_symbol in xai_dict:
            st.bar_chart(xai_dict[selected_symbol], use_container_width=True)
            
    st.divider()
    
    # 자산별 동적 뉴스 피드 매핑 리포트
    st.markdown(f"📰 **[{selected_symbol}] 관련 뉴스 피드 및 감성 지표 실시간 연동**")
    if not df_news.empty:
        # USDT 접미사를 제거하여 매칭 (예: BTCUSDT -> BTC)
        base_coin = selected_symbol.replace("USDT", "")
        coin_news = df_news[df_news['coin'] == base_coin]
        
        if not coin_news.empty:
            for _, row in coin_news.head(5).iterrows():
                if row['pos'] > row['neg']:
                    badge = "🟢 Positive"
                elif row['neg'] > row['pos']:
                    badge = "🔴 Negative"
                else:
                    badge = "🟡 Neutral"
                
                st.markdown(f"- [{row['title']}]({row['url']}) | {badge} | `Score: {row['score']}` (출처: {row['source']})")
        else:
            st.info("해당 자산과 매핑된 최신 감성분석 뉴스 데이터가 없습니다.")
    else:
        st.info("뉴스 감성분석 데이터셋 파일이 비어있거나 읽어오지 못했습니다.")

# ==========================================
# 탭 2: 대화형 에이전트 (RAG 적용)
# ==========================================
with tab2:
    st.subheader("🗣️ 뉴스 감성 지표 기반 투자 전략 상담실")
    st.caption("TFT 모델의 수치 예측 결과와 정성적 뉴스 감성 스코어를 결합하여 맞춤형 브리핑을 제공합니다.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("질문 (예: 비트코인의 매수 적합성을 최근 뉴스 감성 지표와 엮어서 분석해줘)"):
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        # RAG 검색 로직: 질문과 관련된 뉴스 내용 및 긍/부정 비율을 임베딩 검색
        context_str = "참조할 최신 뉴스 컨텍스트가 존재하지 않습니다."
        if vectorstore is not None:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            relevant_docs = retriever.invoke(user_question)
            context_str = "\n".join([doc.page_content for doc in relevant_docs])
        
        # TFT 추론 결과를 요약 텍스트로 변환하여 프롬프트에 제공
        portfolio_summary_lines = []
        for _, row in summary_df.head(3).iterrows():
            portfolio_summary_lines.append(
                f"- {row['Symbol']}: {row['Strategy_Signal']} 시그널 (최적 매수:{row['Best_Buy_Timing']}, 최적 매도:{row['Best_Sell_Timing']}, 예상수익률:{row['Expected_Return_Pct']:.2f}%)"
            )
        portfolio_str = "\n".join(portfolio_summary_lines)

        prompt_template = PromptTemplate(
            input_variables=["portfolio_data", "context", "question"],
            template="""너는 TFT 계량 모형과 뉴스 감성 정성 지표를 결합 분석하는 크립토 전문 퀀트 에이전트야.
            
            [TFT 수치 모형 결과]
            {portfolio_data}
            
            [실제 뉴스 감성 스코어 검색 결과 (RAG)]
            {context}
            
            사용자 질문: {question}
            
            지시사항: 위의 [TFT 수치 모형 결과]의 매매 타이밍 정보와 [RAG 뉴스 감성 지표]의 점수(긍정, 부정 비율 등)를 정교하게 융합하여 정량적/정성적 균형이 잡힌 리포트를 친절한 한글로 작성해줘. 
            """
        )

        formatted_prompt = prompt_template.format(
            portfolio_data=portfolio_str, 
            context=context_str, 
            question=user_question
        )

        with st.chat_message("assistant"):
            with st.spinner("임베딩 지식베이스에서 감성 지표 데이터 탐색 중..."):
                try:
                    response = llm.invoke(formatted_prompt)
                except Exception as ex:
                    response = f"LLM 구동 에러가 발생했습니다: {ex}. 로컬 Ollama 프로세스가 활성화되어 있는지 확인하세요."
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})