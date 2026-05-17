# conda activate gcy6902_llm
# cd btc_llm_test
# python -m streamlit run app.py
import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

st.set_page_config(layout="wide", page_title="AI 트레이딩 시스템 (RAG + 대시보드)")

llm = Ollama(model="qwen2.5:3b")
embeddings = OllamaEmbeddings(model="qwen2.5:3b")

summary_data = [
    {"Symbol": "BTCUSDT", "Decision": "🔴 강한 매수", "Score": 92, "Expected": "+4.5%"},
    {"Symbol": "SOLUSDT", "Decision": "🔴 매수", "Score": 85, "Expected": "+2.1%"},
    {"Symbol": "ETHUSDT", "Decision": "🟡 보합", "Score": 55, "Expected": "+0.2%"},
]
summary_df = pd.DataFrame(summary_data)

dates = pd.date_range(start="2025-01-01", periods=22, freq="8h")
detail_data = {
    "BTCUSDT": {
        "prices": [12, 13, 13.8, 14.1, 14.6, 18.1, 18.2, 18.8, 19.6, 14.8, 16.0, 15.0, 15.5, 15.8, 22.6, 22.9, 23.7, 21.5, 22.1, 20.6, 16.6, 17.4],
        "xai": {"Close": 0.45, "Low": 0.18, "Taker buy": 0.15, "High": 0.07, "Open": 0.05}
    },
    "SOLUSDT": {
        "prices": [20, 22, 21, 25, 28, 30, 31, 29, 35, 36, 38, 40, 42, 41, 45, 48, 50, 49, 52, 55, 58, 60],
        "xai": {"Volume": 0.38, "Close": 0.22, "Taker Buy": 0.18, "M2": 0.12}
    },
    "ETHUSDT": {
        "prices": [10, 11, 10.5, 11.2, 11.5, 12, 11.8, 12.5, 13, 12.8, 13.5, 14, 13.8, 14.5, 15, 14.8, 15.5, 16, 15.8, 16.5, 17, 16.8],
        "xai": {"Open Interest": 0.30, "RSI": 0.25, "Low": 0.20, "Close": 0.15}
    }
}

compare_dict = {"Date": dates}
top_symbols = [d["Symbol"] for d in summary_data]

for sym in top_symbols:
    raw_prices = detail_data[sym]["prices"]
    base_price = raw_prices[0]
    pct_changes = [((p - base_price) / base_price) * 100 for p in raw_prices]
    compare_dict[sym] = pct_changes

compare_df = pd.DataFrame(compare_dict).set_index("Date")


news_texts = [
    "2025년 1월 20일 뉴스: 비트코인이 전 세계적인 기관 자금 유입에 힘입어 22K를 돌파했습니다. 특히 미국 월스트리트의 주요 펀드들이 암호화폐 비중을 확대하고 있습니다.",
    "2025년 1월 21일 뉴스: 나스닥 지수가 3일 연속 상승하며 기술주와 동조화 현상을 보이는 암호화폐 시장도 전반적인 상승 랠리를 시작할 가능성이 높습니다.",
    "2025년 1월 22일 뉴스: 미국 연방준비제도(Fed)가 금리를 동결할 것이라는 기대감이 커지면서, 크립토 시장의 펀딩비가 안정화되고 매수 심리가 회복되고 있습니다."
]
news_links = [
    {"title": "비트코인, 기관 자금 유입에 22K 돌파... 상승 랠리 시작되나?", "url": "https://www.coindesk.com/"},
    {"title": "나스닥 훈풍에 암호화폐 시장 동반 상승세", "url": "https://kr.investing.com/"},
    {"title": "미 연준 금리 동결 유력, 크립토 펀딩비 하락 안정화", "url": "https://cointelegraph.com/"}
]

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
    
    [최신 시장 뉴스 (RAG 검색 결과)]
    {context}
    
    사용자의 질문: {question}
    
    위 'AI 모델 예측' 수치와 '최신 시장 뉴스'를 종합적으로 분석해서 답변해. 특히 뉴스의 내용을 구체적으로 언급하며 매수/매도 판단 근거를 설명해줘.
    """
)

st.title("🤖 AI 암호화폐 트레이딩 시스템 (대시보드 + RAG)")

tab1, tab2 = st.tabs(["📊 요약 대시보드", "💬 대화형 에이전트"])

with tab1:
    st.subheader("💡 AI 모델 종합 종목 추천 (Top 3 추출)")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("**📈 Top 3 종목 누적 변동률(%) 비교**")
    st.info("기준일(Day 1) 가격을 0%로 설정하여, 각 종목의 상대적인 모멘텀을 비교합니다.")
    st.line_chart(compare_df, use_container_width=True)

    st.divider()

    st.subheader("🔍 개별 심볼 상세 분석 및 판단 근거")
    selected_symbol = st.selectbox("분석할 심볼을 선택하세요:", summary_df["Symbol"].tolist())
    
    current_prices = detail_data[selected_symbol]["prices"]
    trend_df = pd.DataFrame({"Date": dates, "Price": current_prices}).set_index("Date")
    
    xai_data = detail_data[selected_symbol]["xai"]
    reasoning_df = pd.DataFrame({"요인": list(xai_data.keys()), "기여도": list(xai_data.values())}).set_index("요인")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**1. {selected_symbol} 절대 가격 트렌드 ($)**")
        st.line_chart(trend_df, use_container_width=True)
    with col2:
        st.markdown(f"**2. {selected_symbol} 판단 근거 (Feature Importance)**")
        st.bar_chart(reasoning_df, use_container_width=True)
        
    st.divider()
    
    st.markdown("📰 **실시간 주요 외부 뉴스 및 시황 (RAG 참조 데이터)**")
    for news in news_links:
        st.markdown(f"- [{news['title']}]({news['url']})")

with tab2:
    st.subheader("🗣️ 뉴스 기반 투자 전략 상담")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_question := st.chat_input("질문 (예: Top 추천 코인과 뉴스를 연관지어 설명해줘)"):
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})
    
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        relevant_docs = retriever.invoke(user_question)
        context_str = "\n".join([doc.page_content for doc in relevant_docs])
        
        portfolio_str = "\n".join([f"- {i['Symbol']}: {i['Decision']} (신뢰도 {i['Score']}%)" for i in summary_data])
    
        formatted_prompt = prompt_template.format(
            portfolio_data=portfolio_str, 
            context=context_str, 
            question=user_question
        )
    
        with st.chat_message("assistant"):
            with st.spinner("관련 뉴스를 검색하고 데이터를 분석 중입니다..."):
                response = llm.invoke(formatted_prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})