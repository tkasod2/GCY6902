import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# 1. LLM 두뇌 연결 (확인된 qwen2.5:3b 모델 사용)
llm = Ollama(model="qwen2.5:3b")

# 2. 모델 예측 데이터
dummy_portfolio = [
    {"Symbol": "BTCUSDT", "decision": "상승", "confidence_score": 0.85, "return": "+2.5%"},
    {"Symbol": "ETHUSDT", "decision": "보합", "confidence_score": 0.55, "return": "+0.2%"},
    {"Symbol": "SOLUSDT", "decision": "강한 상승", "confidence_score": 0.92, "return": "+4.5%"}
]

# 3. 에이전트의 역할과 제약사항을 정의하는 시스템 프롬프트
prompt_template = PromptTemplate(
    input_variables=["portfolio_data", "question"],
    template="""너는 객관적인 데이터를 바탕으로 암호화폐 포트폴리오를 관리하는 전문 트레이딩 에이전트야.

    [현재 AI 모델의 종목별 예측 데이터]
    {portfolio_data}

    사용자의 질문: {question}

    위 데이터를 바탕으로 어떤 종목을 매수하는 것이 가장 합리적일지 비교 분석해서 답변해줘.
    가장 신뢰도(confidence_score)와 예상 수익이 높은 종목을 'Top Pick'으로 추천하고 그 이유를 설명해줘.
    환각(거짓 정보)을 만들지 말고, 제공된 데이터 안에서만 대답해.
    """
)
# 4. 화면 UI 구성
st.title("🤖 AI 암호화폐 트레이딩 에이전트")

# 시각화 대시보드 (판단 근거)
with st.sidebar:
    st.write("📊 **현재 시장 예측 현황 (비교표)**")
    # 파이썬 데이터를 보기 좋은 표 형식으로 변환
    df = pd.DataFrame(dummy_portfolio)
    # 0.85 같은 숫자를 85% 로 예쁘게 보이기
    df['confidence_score'] = df['confidence_score'].apply(lambda x: f"{int(x*100)}%") 
    # 인덱스(0, 1, 2) 숨기고 깔끔하게 출력
    st.dataframe(df, hide_index=True)

# 세션 상태에 채팅 기록 저장
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 채팅 기록 화면에 그리기
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 질문 입력 창
if user_question := st.chat_input("질문 (예: 지금 어떤 코인을 사는 게 가장 좋아?)"):
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # 데이터를 문자열로 예쁘게 조립해서 프롬프트에 전달 준비
    portfolio_str = "\n".join([f"- {item['Symbol']}: 예측({item['decision']}), 신뢰도({item['confidence_score']}), 예상수익({item['return']})" for item in dummy_portfolio])

    formatted_prompt = prompt_template.format(
        portfolio_data=portfolio_str,
        question=user_question
    )

    # LLM 답변 생성 및 출력
    with st.chat_message("assistant"):
        with st.spinner("여러 종목을 비교 분석 중입니다..."):
            response = llm.invoke(formatted_prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})