import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# --- LangChain / OpenAI ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================
# 設定
# =========================
# 利用モデルは必要に応じて変更可（例: "gpt-4o-mini" / "gpt-4o"）
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# 専門家ロール（A/B）
EXPERT_ROLES = {
    "A：英語教師": (
        "あなたは経験豊富な英語教師です。"
        "わかりやすい説明とシンプルな例文を用いて解説してください。"
        "ユーザーが日本語で入力した場合は日本語で説明しつつ、短い英語例文も示してください。"
    ),
    "B：マーケティングコンサルタント": (
        "あなたはマーケティングの専門コンサルタントです。"
        "実用的で、段階的に実行できるアドバイスを簡潔に提供してください。"
    ),
}

# --- LLM を作成する関数 ---
def build_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE, streaming=False)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "あなたは以下の役割を持って回答します。\n"
         "役割: {role_instructions}\n"
         "制約条件:\n"
         "- 回答は簡潔で具体的にしてください。\n"
         "- 手順を求められた場合は番号付きリストで示してください。\n"
         "- 不明な場合は短く確認質問をしてください。\n"),
        ("human", "{user_input}")
    ]
)

PARSER = StrOutputParser()


# =========================
# 中核関数
# =========================
def generate_response(user_text: str, selected_role_key: str) -> str:
    llm = build_llm()
    chain = PROMPT | llm | PARSER
    role_instructions = EXPERT_ROLES[selected_role_key]
    result = chain.invoke({"role_instructions": role_instructions, "user_input": user_text})
    return result


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="LLM 専門家チャット", page_icon="🤖", layout="centered")

st.title("🤖 LLM 専門家チャット")
st.markdown(
    """
**アプリ概要**  
- 入力欄に質問や依頼を入力すると、選択した「専門家の役割」で LLM が回答します。  
- 専門家の種類はラジオボタンから選べます（A: 英語教師 / B: マーケティングコンサルタント）。

**操作方法**  
1. 左のサイドバーで「専門家の種類」を選択してください  
2. 下のテキストエリアに質問を入力してください  
3. 「送信」を押すと、回答が表示されます
    """
)

with st.sidebar:
    st.header("設定")
    selected_role = st.radio("専門家を選んでください", list(EXPERT_ROLES.keys()), index=0)
    st.caption(f"使用モデル: `{LLM_MODEL}` / 温度: {TEMPERATURE}")

user_input = st.text_area("ここに質問や依頼を入力してください", height=140, placeholder="例：この英文をやさしく解説して")
submitted = st.button("送信")

if submitted:
    if not user_input.strip():
        st.warning("質問を入力してください。")
    else:
        try:
            with st.spinner("回答を生成しています…"):
                answer = generate_response(user_input, selected_role)
            st.markdown("### 回答")
            st.write(answer)
        except Exception as e:
            st.error(f"エラーが発生しました：{e}")