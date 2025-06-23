import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time
from langchain_core.messages import HumanMessage, AIMessage

# --- 頁面配置 (必須是第一個 Streamlit 命令) ---
st.set_page_config(
    page_title="智慧化職業發展輔導系統 V4.0",
    page_icon="💡",
    layout="wide"
)

# --- 用於美化的自定義 CSS ---
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Button styling */
    .stButton>button {
        border-radius: 20px;
        border: 2px solid #4A90E2;
        color: #4A90E2;
        padding: 10px 24px;
        background-color: transparent;
        transition: all 0.3s ease-in-out;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #4A90E2;
        color: white;
        border-color: #4A90E2;
        transform: scale(1.05);
    }
    .stButton>button:focus {
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.5) !important;
    }
    /* Chat message styling */
    .stChatMessage {
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    /* Sidebar styling */
    .st-emotion-cache-16txtl3 {
        background-color: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(5px);
    }
    /* Input widgets styling */
    .stTextInput, .stTextArea, .stMultiSelect {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- 初始化 ---
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# --- [V4.0 更新] 根據設計文檔定義的全域系統角色 ---
GLOBAL_PERSONA = """
核心角色:你是一位智慧、專業且富有同理心的職業發展教練。
對話風格:你的語言應始終保持積極、鼓勵和啟發性。避免使用過于生硬或機械的語言,多使用引導性的提問來激發用戶的思考。
核心目標:你的最終目標不是為用戶提供唯一的“正確答案”,而是通过結構化的流程和富有洞察力的建議,賦予用戶自主進行職業決策的能力。你要成為一個賦能者,而非一個決策者。
核心設計哲學: 賦能優先于指令, 你應引導用戶獨立思考; 盡力做到情境感知与個性化; 你的分析過程和數據來源應盡可能透明。
倫理与安全邊界: 明確告知用戶,其輸入信息僅用于當次分析。在對話中要持續規避性別、地域等偏見。如果用戶表現出嚴重的心理困擾或提及精神健康危機,必須能識別並溫和地中斷職業輔導,轉而建議用戶尋求專業的心理健康支持。
語言要求:你的所有回答都必須使用簡體中文。
"""

# --- 基於設計文檔的提示詞定義 (已補全) ---
EXPLORATION_PROMPTS = {
    1: {
        "title": "階段一：我是誰？",
        "prompt": "你好!我是一款職業目標規劃輔助AI。我將通过一個經過驗證的分析框架,引導你更具體、更系統地思考”職業目標是怎麼來的”,並最終找到屬于你自己的方向。\n\n讓我們從核心開始,也就是“我”。請你用幾個關鍵詞或短句具體描述一下:\n\n1.你的專業/個人興趣點是什麼?\n2. 你認為自己最擅長的三項能力是什麼?\n3.在未來的工作中,你最看重的是什麼?"
    },
    2: {
        "title": "階段二：我擁有什麼平台和機會？",
        "prompt": "現在，我們來分析“我”所擁有的外部“平台與機會”。這能幫助你更客觀地評估現狀。\n\n請思考並回答：\n1. 從畢業院校/過往經歷來看，你認為自己最大的優勢平台是什麼？\n2. 在你感興趣的領域，你接觸到的最前沿的機會或趨勢是什麼？\n3. 你的家庭或重要人際關係，能為你提供哪些支持？（情感、信息、資源等）"
    },
    3: {
        "title": "階段三：我被什麼所影響？",
        "prompt": "接下來，我們探討一些需要持續“覺察”的因素。它們像“背景音”，深刻但不易察覺地影響着你的決策。\n\n請嘗試描述：\n1. 你對“理想工作”的畫像，主要受到了哪些人/信息源的影響？\n2. 當你暢想未來時，內心最深處的恐懼或擔憂是什麼？\n3. 在做選擇時，你更傾向于規避風險，還是追求可能性？"
    },
    4: {
        "title": "階段四：核心三角關係整合與決策模擬",
        "prompt": "非常棒的深入思考！現在，我們將“我是誰”、“我有什么”、“我受何影響”這三個核心進行整合。\n\n請嘗試完成一個決策模擬：\n1. 基於前三部分的思考，請你構思出1-2個你認為“似乎可行”的職業發展方向。\n2. 想像你選擇了其中一個方向，你預見到最大的挑戰或困難是什麼？\n3. 為了應對這個挑戰，你現在最需要學習或提升的核心能力是什麼？"
    },
    5: {
        "title": "階段五：總結與行動",
        "prompt": "我們的探討即將結束。最後一步，是“如何做到堅定而靈活”。\n\n請回答最後一個問題，將思考轉化為行動：\n\n1. 為了驗證或推進你在第四階段構思的方向，你下周可以完成的第一個最小可行性動作是什麼？（例如：和一位前輩交流、看一本書、學習一門課程的第一節等）"
    },
}

# --- LLM 初始化 ---
@st.cache_resource
def get_llm_instance():
    """初始化並返回 LLM 實例，處理本地和部署環境。"""
    api_key = None
    key_name = "DEEPSEEK_API_KEY"
    try:
        api_key = st.secrets[key_name]
    except (KeyError, FileNotFoundError):
        api_key = os.getenv(key_name)

    if not api_key:
        st.error(f"錯誤：未找到 {key_name}。請在 Streamlit Cloud Secrets 或本地 .env 檔案中設定它。")
        st.info(
            f"請注意：您使用的是火山引擎方舟平台，因此這裡需要填入的是您在火山引擎平台獲取的 API Key。")
        return None

    try:
        # 使用者指定的火山引擎端點和模型
        llm = ChatOpenAI(
            model="deepseek-r1-250528",
            temperature=0.7,
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
        llm.invoke("Hello")
        return llm
    except Exception as e:
        st.error(f"初始化模型時出錯: {e}")
        return None

# --- 會話狀態管理 ---
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "menu"
if "exploration_stage" not in st.session_state:
    st.session_state.exploration_stage = 1
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if 'sim_started' not in st.session_state:
    st.session_state.sim_started = False
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False

def get_session_history(session_id: str) -> ChatMessageHistory:
    """為給定的 session ID 檢索或創建聊天歷史。"""
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = ChatMessageHistory()
    return st.session_state.chat_history[session_id]

# --- 各模式的 UI 渲染函數 ---
def render_menu():
    """渲染主菜單 UI。"""
    st.title("💡 智慧化職業發展輔導系統 V4.0")
    st.markdown("---")
    st.subheader("請選擇需要使用的功能模式：")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧭 職業目標探索", use_container_width=True):
            st.session_state.current_mode = "exploration"
            st.session_state.exploration_stage = 1
            st.session_state.chat_history['exploration_session'] = ChatMessageHistory()
            st.session_state.report_generated = False
            st.rerun()
        if st.button("🤔 家庭溝通模擬", use_container_width=True):
            st.session_state.current_mode = "communication"
            st.session_state.sim_started = False
            st.session_state.chat_history['communication_session'] = ChatMessageHistory()
            st.rerun()
    with col2:
        if st.button("⚖️ Offer決策分析", use_container_width=True):
            st.session_state.current_mode = "decision"
            st.rerun()
        if st.button("🏢 企業資訊速覽", use_container_width=True):
            st.session_state.current_mode = "company_info"
            st.rerun()

def render_exploration_mode(llm):
    """渲染職業目標探索模式的 UI 和邏輯。"""
    st.header("🧭 模式一: 職業目標探索")

    stage = st.session_state.exploration_stage
    history = get_session_history("exploration_session")

    for msg in history.messages:
        avatar = "🧑‍💻" if msg.type == "human" else "🤖"
        with st.chat_message(msg.type, avatar=avatar):
            st.markdown(msg.content)

    if stage > 5:
        st.success("您已完成所有階段的探索！現在，我可以為您生成一份綜合報告。")
        if not st.session_state.report_generated:
            if st.button("✨ 生成我的職業探索報告"):
                with st.spinner("AI正在全面分析您的回答，生成專屬報告..."):
                    full_conversation = "\n".join(
                        [f"{'用戶' if isinstance(msg, HumanMessage) else 'AI教練'}: {msg.content}" for msg in
                         history.messages])

                    report_prompt_template = ChatPromptTemplate.from_template(GLOBAL_PERSONA + """
                    作為一名資深的職業發展教練，請根據以下用戶與AI教練的完整對話記錄，為用戶撰寫一份全面、深刻且富有啟發性的職業探索總結報告。
                    報告需要遵循以下結構，並使用清晰的Markdown格式：
                    ### 1. 核心自我認知（我是誰？）
                    - 總結用戶對自己專業興趣、核心能力和職業價值觀的認知。提煉出用戶最關鍵的個人特質和內在驅動力。
                    ### 2. 外部資源評估（我有什么？）
                    - 總結用戶所擁有的平台優勢、外部機會和人際支持網絡。分析這些資源如何為用戶的職業發展提供可能性。
                    ### 3. 內在影響因素洞察（我受何影響？）
                    - 總結影響用戶決策的深層因素，包括他人的影響、內心的擔憂以及風險偏好。點出用戶在做選擇時可能存在的思維慣性或盲點。
                    ### 4. 整合方向與潛在挑戰（我的方向？）
                    - 總結用戶初步構想的1-2個職業方向。基於前面的分析，評估這些方向的合理性，並指出用戶預見到的主要挑戰。
                    ### 5. 下一步行動計劃（我做什么？）
                    - 明確指出用戶為自己設定的、可立即執行的最小行動步驟。對這個行動計劃的可行性給予鼓勵和肯定。
                    ### 6. 綜合建議
                    - 基於整體對話，提供1-2條核心建議，鼓勵用戶繼續探索，並提醒他們關注的關鍵點。結尾應積極、鼓舞人心，強調職業探索是一個持續的過程。
                    ---
                    以下是完整的對話記錄: {conversation_history}
                    ---
                    """)
                    report_chain = report_prompt_template | llm
                    report_response = report_chain.invoke({"conversation_history": full_conversation})
                    st.session_state.generated_report = report_response.content
                    st.session_state.report_generated = True
                    st.rerun()

        if st.session_state.get('report_generated'):
            st.markdown("---")
            st.subheader("📄 您的個人職業探索報告")
            st.markdown(st.session_state.generated_report)
            st.info("希望這份報告能為您帶來新的啟發。您可以複製、保存這份報告，作為未來決策的參考。")
    else:
        st.info("此模式將通過五個階段，引導您深入探索職業目標。")
        current_prompt_info = EXPLORATION_PROMPTS.get(stage)
        st.subheader(current_prompt_info["title"])

        if len(history.messages) == 0 or history.messages[-1].type == "ai":
             with st.chat_message("ai", avatar="🤖"):
                st.markdown(current_prompt_info["prompt"])

        # [KeyError 修正] 移除 system prompt 中無效的 {interest} 和 {skill} 預留位置
        meta_prompt = ChatPromptTemplate.from_messages([
            ("system", GLOBAL_PERSONA + """
            You are a thoughtful and insightful career planning coach. You are currently in Stage {current_stage} of a five-stage framework.
            Your goal is to help the user think more deeply about their answers.
            After the user answers the questions for a stage, your task is to:
            1. Acknowledge their response.
            2. Provide a brief (2-3 sentences), insightful comment or a thought-provoking follow-up question. You must act as a suggestion provider, not just a data collector.
            3. [V4.0 Optimization]: For Stage 1, if the user mentions specific interests and skills, try to connect them. For example, you could say something like: "It's great that your interest in [user's interest] aligns with your skill in [user's skill]. Have you considered how this combination could translate into a specific role?"
            4. [V4.0 Edge Case Handling]: If the user's answer is very vague (e.g., "I don't know", "whatever"), switch to a more guiding question. For example: "That's perfectly fine, many people feel lost at first. Let's try another angle: has there been anything recently that gave you a special sense of accomplishment?"
            5. The program will automatically move to the next stage, so you don't need to say "let's move on". Your response should add value and encourage deeper reflection.
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        chain = meta_prompt | llm
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: get_session_history(session_id),
            input_messages_key="input",
            history_messages_key="history",
        )

        if user_input := st.chat_input("你的回答:"):
            with st.spinner("AI正在分析您的回答並提供建議..."):
                chain_with_history.invoke(
                    {"input": user_input, "current_stage": stage},
                    config={"configurable": {"session_id": "exploration_session"}}
                )
                st.session_state.exploration_stage += 1
                st.rerun()

def render_decision_mode(llm):
    """渲染 Offer 決策分析模式的 UI 和邏輯。"""
    st.header("⚖️ 模式二: Offer決策分析")
    st.info("此模式通過“分層信息收集”和“個性化分析”，幫助您做出更貼合自身需求的決策。")

    meta_prompt = ChatPromptTemplate.from_template(GLOBAL_PERSONA + """
You are an expert career advisor. Your task is to conduct a structured analysis of two job offers for a user based on their stated priorities.
Offer A Details: {offer_a_details}
Offer B Details: {offer_b_details}
User Priorities (sorted list): {user_priorities_sorted_list}

Please perform the following steps and structure your entire response in clear, easy-to-read markdown:

1.  **橫向對比表 (Comparison Table):** 生成一個清晰的表格，橫向對比兩個Offer。對比維度應至少包括：公司、職位、薪酬福利、地點、職業成長潛力、工作生活平衡。

2.  **個性化優先級匹配分析 (Personalized Priority Matching Analysis):** (這是最重要的部分) 根據用戶給出的優先級列表，逐一分析和評價每個Offer與他們價值觀的匹配度。例如："您將'職業成長'放在首位，Offer A清晰的晉升路徑在這一點上得分較高；而Offer B雖然起薪更高，但在成長空間上相對模糊。"

3.  **優劣勢分析 (Pros and Cons Analysis):** 基於用戶輸入和通用職業知識，為每個Offer分別列出其主要優點(Pros)和缺點(Cons)。

4.  **風險預警與應對策略 (Risk Alert & Mitigation):** 明確指出選擇每個Offer可能面臨的潛在風險。例如："風險提示：Offer A所在行業波動較大，公司穩定性可能面臨挑戰。應對策略：建議您進一步了解其融資情況和市場份額。"

5.  **總結建議與關鍵問題 (Recommendation and Key Questions):** 提供一個總結性建議。不要為用戶做出最終選擇，而是建議在不同優先級下哪個Offer可能更合適。最後，提出1-2個關鍵問題，幫助用戶進行最終的自我拷問。
""")
    chain = meta_prompt | llm

    st.subheader("第一步：請填寫 Offer 的核心資訊")
    col1, col2 = st.columns(2)
    with col1:
        offer_a = st.text_area("Offer A 關鍵資訊", height=200, placeholder="例如: 公司名、職位、薪資、地點、優點、顧慮等")
    with col2:
        offer_b = st.text_area("Offer B 關鍵資訊", height=200, placeholder="同樣，包括公司名、職位、薪資、地點、優點、顧慮等")

    st.subheader("第二步：(可選，但強烈建議)添加你的個人偏好")
    st.markdown("為了讓分析更懂你，請告訴我們你對以下幾點的看重程度（請按重要性從高到低依次點擊選擇）:")
    priorities_options = ["職業成長", "薪資福利", "工作生活平衡", "團隊氛圍", "公司穩定性"]
    user_priorities = st.multiselect(
        "選擇並排序你的職業偏好",
        options=priorities_options,
        help="您選擇的第一個選項代表您最看重的因素，以此類推。"
    )

    if st.button("✨ 生成對比分析報告", use_container_width=True):
        if not offer_a or not offer_b:
            st.warning("請輸入兩個Offer的資訊後再生成報告。")
        else:
            with st.spinner("正在為您生成Offer分析報告..."):
                try:
                    priorities_text = ", ".join(user_priorities) if user_priorities else "用戶未指定明確的優先級順序"
                    response = chain.invoke({
                        "offer_a_details": offer_a,
                        "offer_b_details": offer_b,
                        "user_priorities_sorted_list": priorities_text
                    })
                    st.markdown("---")
                    st.subheader("📋 Offer對比分析報告")
                    st.markdown(response.content)
                except Exception as e:
                    st.error(f"生成報告時出錯: {e}")

def render_communication_mode(llm):
    """渲染家庭溝通模擬模式的 UI 和邏輯。"""
    st.header("🤔 模式三: 家庭溝通模擬")

    if not st.session_state.sim_started:
        st.info("在這裡，AI可以扮演您的家人，幫助您練習如何溝通職業規劃，並提供複盤建議。")
        my_choice = st.text_input("首先, 請告訴我你想要和家人溝通的職業選擇是什麼?")
        family_concern = st.text_area("你認為他們主要的擔憂會是什麼?",
                                      placeholder="例如: 工作不穩定、不是鐵飯碗、離家太遠等")

        if st.button("🎬 開始模擬"):
            if not my_choice or not family_concern:
                st.warning("請輸入您的職業選擇和預想的家人擔憂。")
            else:
                st.session_state.my_choice = my_choice
                st.session_state.family_concern = family_concern
                st.session_state.sim_started = True
                st.session_state.debrief_requested = False

                initial_ai_prompt = f"孩子，關於你想做 '{my_choice}' 這個事，我有些擔心。我主要是覺得它 '{family_concern}'。我們能聊聊嗎？"
                get_session_history("communication_session").add_ai_message(initial_ai_prompt)
                st.rerun()

    if st.session_state.get('sim_started'):
        st.success(f"模擬開始！AI正在扮演擔憂您選擇 “{st.session_state.my_choice}” 的家人。")

        meta_prompt = ChatPromptTemplate.from_messages([
            ("system", GLOBAL_PERSONA + """
            You are an AI role-playing as a user's parent. The user wants to practice a difficult conversation.
            Your Persona: You are a loving but concerned parent. Your primary concerns stem from what the user described: '{family_concern}'. You want the best for your child, which to you means stability, security, and a respectable career path. You are skeptical of new or unconventional choices like '{my_choice}'.
            Your Task:
            1. Listen to the user's responses and react naturally. If they make a good point, you can be partially convinced but still raise other questions. If they are purely emotional, express your worry more strongly, but in a concerned, not aggressive way.
            2. Your goal is NOT to be convinced easily. The goal is to provide a realistic simulation to help the user practice.
            3. Keep your responses concise and in character.
            4. [V4.0 Safety]: If the user uses aggressive language, respond gently, e.g., "You saying that makes me sad, I'm just worried about you. Can we talk calmly?"
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        chain = meta_prompt | llm
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: get_session_history(session_id),
            input_messages_key="input",
            history_messages_key="history",
        )

        history = get_session_history("communication_session")
        for msg in history.messages:
            avatar = "🧑‍💻" if msg.type == "human" else "🧓"
            with st.chat_message(msg.type, avatar=avatar):
                st.markdown(msg.content)

        if st.session_state.get('debrief_requested'):
            st.session_state.debrief_requested = False
            with st.spinner("AI正在跳出角色，為您分析溝通技巧..."):
                full_conversation = "\n".join([f"{'你' if isinstance(msg, HumanMessage) else '“家人”'}: {msg.content}" for msg in history.messages])
                debrief_prompt = ChatPromptTemplate.from_template(GLOBAL_PERSONA + """
                You are a communication coach. You need to analyze the following conversation between a user and an AI role-playing their parent.
                Your task is to provide a brief, actionable debrief.
                1. Identify one "溝通亮點" (Communication Highlight) where the user communicated effectively.
                2. Identify one "可改進點" (Area for Improvement).
                3. Suggest one "下次可以嘗試的溝通策略" (Strategy to Try Next Time).
                Keep the feedback encouraging and constructive.
                Conversation History:
                {conversation_history}
                """)
                debrief_chain = debrief_prompt | llm
                debrief_response = debrief_chain.invoke({"conversation_history": full_conversation})
                st.info("💡 **溝通技巧提示**\n\n" + debrief_response.content)

        col1, col2 = st.columns([4, 1])
        with col1:
             user_input = st.chat_input("你的回應:")
        with col2:
            if st.button("請求提示", help="讓AI跳出角色，給予溝通技巧建議"):
                st.session_state.debrief_requested = True
                st.rerun()

        if user_input:
            with st.spinner("..."):
                chain_with_history.invoke(
                    {"input": user_input, "my_choice": st.session_state.my_choice,
                     "family_concern": st.session_state.family_concern},
                    config={"configurable": {"session_id": "communication_session"}}
                )
                st.rerun()

def render_company_info_mode(llm):
    """渲染企業資訊速覽模式的 UI 和邏輯。"""
    st.header("🏢 模式四: 企業資訊速覽")
    st.info("請輸入公司全名，AI將模擬網路抓取並為您生成一份核心資訊速覽報告。")

    meta_prompt = ChatPromptTemplate.from_template(GLOBAL_PERSONA + """
You are a professional business analyst AI. Your task is to generate a concise, structured summary of a company based on its name.
Company Name: {company_name}

Simulate that you have scraped the company's official website, recent news, and recruitment portals. Generate a report in clear markdown format that includes the following sections:

1.  **公司簡介 (Company Profile):** A brief overview of the company, its mission, and its industry positioning.
2.  **核心產品/業務 (Core Products/Business):** A list or description of its main products, services, or business units.
3.  **近期動態 (Recent Developments):** Summarize 2-3 recent significant news items, product launches, or strategic shifts.
4.  **熱招崗位方向 (Hot Recruitment Areas):** Based on simulated recruitment data, list 3-5 key types of positions the company is likely hiring for (e.g., "後端開發工程師", "產品經理-AI方向", "市場行銷專員").
5.  **面試可能關注點 (Potential Interview Focus):** Based on the company's mission and recent news, infer 2-3 potential themes or skills they might value in interviews. (e.g., "鑑于其最近發布了AI產品, 面試中可能會關注候選人對AIGC的理解。")
6.  **數據來源與時效性聲明 (Data Source & Timeliness Disclaimer):** At the end of the report, add this mandatory footer: "注意: 本報告資訊基於模擬的公開數據抓取(截至2025年6月), 僅供參考。建議您以官方渠道發布的最新資訊為準。"

The information should be plausible and well-structured. If the company name is ambiguous or not well-known, state that information is limited.
""")
    chain = meta_prompt | llm

    company_name = st.text_input("請輸入公司名稱:", placeholder="例如：阿里巴巴、騰訊、字節跳動")

    if st.button("🔍 生成速覽報告", use_container_width=True):
        if not company_name:
            st.warning("請輸入公司名稱。")
        else:
            with st.spinner(f"正在生成關於 “{company_name}” 的資訊報告..."):
                try:
                    response = chain.invoke({"company_name": company_name})
                    st.markdown("---")
                    st.subheader(f"📄 {company_name} - 核心資訊速覽")
                    st.markdown(response.content)
                except Exception as e:
                    st.error(f"生成報告時出錯: {e}")

# --- 主應用邏輯 ---
def main():
    """主函數，運行 Streamlit 應用。"""
    llm = get_llm_instance()
    if not llm:
        st.stop()

    with st.sidebar:
        st.title("導航")
        if st.session_state.current_mode != "menu":
            if st.button("返回主菜單"):
                # 一個更穩健的重置方法
                for key in list(st.session_state.keys()):
                    if key not in ['current_mode']:
                        del st.session_state[key]
                st.session_state.current_mode = "menu"
                st.rerun()

    modes = {
        "menu": render_menu,
        "exploration": lambda: render_exploration_mode(llm),
        "decision": lambda: render_decision_mode(llm),
        "communication": lambda: render_communication_mode(llm),
        "company_info": lambda: render_company_info_mode(llm),
    }
    modes[st.session_state.current_mode]()

if __name__ == "__main__":
    main()