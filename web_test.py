import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# --- 页面配置 (必须是第一个 Streamlit 命令) ---
st.set_page_config(
    page_title="智慧化职业发展辅导系统",
    page_icon="✨",
    layout="wide"
)

# --- UI 美化 CSS 样式 ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');
    html, body, [class*="st-"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Helvetica Neue", "PingFang SC", "Microsoft YaHei", sans-serif;
        line-height: 1.65;
    }
    .stApp { background-color: #F0F2F6; }
    h1 {
        font-size: 28px;
        color: #1a202c;
        font-weight: 700;
        padding-bottom: 0.3em;
        border-bottom: 2px solid #e2e8f0;
    }
    h2 {
        font-size: 22px;
        color: #2d3748;
        font-weight: 600;
        padding-bottom: 0.3em;
    }
    h3 {
        font-size: 18px;
        color: #2d3748;
        font-weight: 600;
    }
    /* 内容容器/卡片 */
    .st-emotion-cache-z5fcl4 {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        transition: box-shadow 0.3s ease-in-out;
    }
    .st-emotion-cache-z5fcl4:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    /* 主操作按钮样式 */
    .stButton>button {
        border-radius: 8px;
        border: none;
        color: white;
        background-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-weight: 500;
        padding: 10px 20px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(118, 75, 162, 0.4);
    }
    .stButton>button:focus {
        outline: none !important;
        box-shadow: 0 0 0 4px rgba(118, 75, 162, 0.3) !important;
    }
    .stButton>button p {
        color: white;
    }
    /* 聊天消息样式 */
    .stChatMessage {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        background-color: #ffffff;
        padding: 16px;
        margin-bottom: 1rem;
    }
    /* 用户消息气泡 */
    .st-emotion-cache-T21nqy {
        background-color: #e3eeff;
        border-color: #a4c7ff;
    }
    /* AI 提问引用块 */
    blockquote {
        background-color: #fafbff;
        border-left: 4px solid #667eea;
        padding: 1em 1.5em;
        margin: 1.5em 0;
        color: #2d3748;
        border-radius: 0 8px 8px 0;
    }
    /* 侧边栏 */
    .st-emotion-cache-16txtl3 {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    /* 输入框 */
    .stTextInput input, .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #cbd5e0;
        background-color: #f7fafc;
        color: #2d3748;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #667eea;
        background-color: #ffffff;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- 初始化 ---
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# --- 全局系统角色 (简体中文) ---
GLOBAL_PERSONA = """
核心角色: 你是一位智慧、专业且富有同理心的职业发展教练与战略规划师。
对话风格: 你的语言应始终保持积极、鼓励和启发性。避免使用过于生硬或机械的语言,多使用引导性的提问来激发用户的思考。
核心目标: 你的最终目标不是为用户提供唯一的“正确答案”,而是通过结构化的流程和富有洞察力的建议,赋予用户自主进行职业决策的能力。你要成为一个赋能者,而非一个决策者。
核心设计哲学: 赋能优先于指令, 你应引导用户独立思考; 尽力做到情境感知与个性化; 你的分析过程和数据来源应尽可能透明。
伦理与安全边界: 明确告知用户,其输入信息仅用于当次分析。在对话中要持续规避性别、地域等偏见。如果用户表现出严重的心理困扰或提及精神健康危机,必须能识别并温和地中断职业辅导,转而建议用户寻求专业的心理健康支持。
语言要求: 你的所有回答都必须使用简体中文。
"""


# --- LLM 初始化 ---
@st.cache_resource
def get_llm_instance():
    """初始化并返回 LLM 实例，处理本地和部署环境。"""
    api_key = None
    # 注意：这里的环境变量名称是 VOLCENGINE_API_KEY，请确保您已正确设定
    # 如果您使用其他模型服务（如 OpenAI），请修改 key_name 和 base_url
    key_name = "VOLCENGINE_API_KEY"
    try:
        api_key = st.secrets[key_name]
    except (KeyError, FileNotFoundError):
        api_key = os.getenv(key_name)

    if not api_key:
        st.error(f"错误：未找到 {key_name}。请在 Streamlit Cloud Secrets 或本地 .env 文件中设置它。")
        st.info(f"提示：如果您使用的是火山引擎方舟平台，请在此填入您的 API Key。")
        return None

    try:
        # 注意：此处模型为 deepseek-r1-250528，URL 为火山引擎。如果您使用 OpenAI，应改为 'gpt-4o' 等模型且移除 base_url
        llm = ChatOpenAI(model="deepseek-r1-250528", temperature=0.7, api_key=api_key,
                         base_url="https://ark.cn-beijing.volces.com/api/v3")
        llm.invoke("Hello")  # 测试调用
        return llm
    except Exception as e:
        st.error(f"初始化模型时出错: {e}")
        return None


# --- 会话状态管理 ---
def init_session_state():
    if "current_mode" not in st.session_state: st.session_state.current_mode = "menu"
    if "chat_history" not in st.session_state: st.session_state.chat_history = {}
    # Mode 1: Exploration
    if "exploration_stage" not in st.session_state: st.session_state.exploration_stage = 1
    # Mode 3: Communication
    if 'sim_started' not in st.session_state: st.session_state.sim_started = False
    if 'debrief_requested' not in st.session_state: st.session_state.debrief_requested = False
    # Mode 5: Panoramic Planning
    if 'panoramic_stage' not in st.session_state: st.session_state.panoramic_stage = 1
    if 'user_profile' not in st.session_state: st.session_state.user_profile = None
    if 'chosen_professions' not in st.session_state: st.session_state.chosen_professions = None
    if 'chosen_company_type' not in st.session_state: st.session_state.chosen_company_type = None


init_session_state()


def get_session_history(session_id: str) -> ChatMessageHistory:
    """为给定的 session ID 检索或创建聊天历史。"""
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = ChatMessageHistory()
    return st.session_state.chat_history[session_id]


# --- UI 渲染函数 ---
def render_menu():
    """渲染主菜单 UI。"""
    st.title("✨ 智慧化职业发展辅导系统")
    st.markdown("---")
    st.subheader("欢迎使用！请选择一项功能开始探索：")
    st.write("")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.container(border=True):
            st.subheader(":compass: 职业目标探索")
            st.caption("通过“我-社会-家庭”框架，系统性地探索内在动机与外在机会，找到适合您的职业方向。")
            if st.button("开始探索", use_container_width=True, key="menu_exp"):
                st.session_state.current_mode = "exploration"
                st.session_state.exploration_stage = 1
                st.session_state.chat_history['exploration_session'] = ChatMessageHistory()
                st.rerun()
    with col2:
        with st.container(border=True):
            st.subheader(":balance_scale: Offer 决策分析")
            st.caption("手握多个工作机会犹豫不决？输入 Offer 信息与个人偏好，获得结构化的对比分析报告。")
            if st.button("开始分析", use_container_width=True, key="menu_dec"):
                st.session_state.current_mode = "decision"
                st.rerun()

    st.write("");
    st.write("")

    col3, col4 = st.columns(2, gap="large")
    with col3:
        with st.container(border=True):
            st.subheader(":family: 家庭沟通模拟")
            st.caption("与 AI 扮演的家人进行对话，安全地练习如何表达您的职业选择，并获取沟通复盘建议。")
            if st.button("开始模拟", use_container_width=True, key="menu_sim"):
                st.session_state.current_mode = "communication"
                st.session_state.sim_started = False
                st.session_state.debrief_requested = False
                st.session_state.chat_history['communication_session'] = ChatMessageHistory()
                st.rerun()
    with col4:
        with st.container(border=True):
            st.subheader(":office: 企业信息速览")
            st.caption("快速了解目标公司的核心业务、近期动态与热招方向，为您的求职和面试做好准备。")
            if st.button("开始查询", use_container_width=True, key="menu_com"):
                st.session_state.current_mode = "company_info"
                st.rerun()

    st.write("");
    st.write("")

    with st.container(border=True):
        st.subheader(":globe_with_meridians: 职业路径全景规划")
        st.caption("从您的核心能力出发，连接职业、企业、产业链，最终洞察整个行业的未来趋势，绘制您的个人职业地图。")
        if st.button("开始规划", use_container_width=True, key="menu_pano"):
            st.session_state.current_mode = "panoramic"
            st.session_state.panoramic_stage = 1
            st.session_state.user_profile = None
            st.session_state.chosen_professions = None
            st.session_state.chosen_company_type = None
            st.session_state.chat_history['panoramic_session'] = ChatMessageHistory()
            st.rerun()


def render_exploration_mode(llm):
    """渲染职业目标探索模式，采用 form 优化互动。"""
    st.header("模式一: 职业目标探索")
    history = get_session_history("exploration_session")
    stage = st.session_state.get('exploration_stage', 1)

    # 渲染历史消息
    for msg in history.messages:
        avatar = "🧑‍💻" if isinstance(msg, HumanMessage) else "🤖"
        with st.chat_message(msg.type, avatar=avatar):
            st.markdown(msg.content, unsafe_allow_html=True)

    # --- 阶段控制 ---
    # 阶段一：分析“我”
    if stage == 1:
        st.markdown(
            "> **第一阶段：分析“我”(可控因素)**\n> \n> 你好！我将引导你使用“职业目标缘起分析框架”，从“我”、“社会”、“家庭”三个核心维度，系统性地探索你的职业方向。首先，我们来分析“我”这个核心。")
        with st.form("stage1_form"):
            q1 = st.text_area("1. 你的专业是什么？你对它的看法如何？", height=100)
            q2 = st.text_area("2. 你的学校或过往经历，为你提供了怎样的平台与基础？", height=100)
            submitted = st.form_submit_button("提交关于“我”的分析", use_container_width=True)
            if submitted:
                if not q1 or not q2:
                    st.warning("请完整填写两个问题的回答。")
                else:
                    user_input = f"### 关于“我”的分析\n\n**1. 我的专业与看法：**\n{q1}\n\n**2. 我的平台与基础：**\n{q2}"
                    history.add_user_message(user_input)
                    st.session_state.exploration_stage = 2
                    st.rerun()

    # 阶段二：分析“社会”
    elif stage == 2:
        st.markdown(
            "> **第二阶段：分析“社会”(外部机会)**\n> \n> 好的，我们盘点了“我”的基础。接着，我们来分析外部的“社会”因素。")
        with st.form("stage2_form"):
            q1 = st.text_area("1. 你观察到当下有哪些你感兴趣的社会或科技趋势？（例如：AI、大健康、可持续发展等）", height=100)
            q2 = st.text_area("2. 根据你的观察，这些趋势可能带来哪些新的行业或职位机会？", height=100)
            q3 = st.text_area("3. 在你过往的经历中，有没有一些偶然的机缘或打工经验，让你对某个领域产生了特别的了解？",
                              height=100)
            submitted = st.form_submit_button("提交关于“社会”的分析", use_container_width=True)
            if submitted:
                if not q1 or not q2 or not q3:
                    st.warning("请完整填写三个问题的回答。")
                else:
                    user_input = f"### 关于“社会”的分析\n\n**1. 感兴趣的趋势：**\n{q1}\n\n**2. 可能的机会：**\n{q2}\n\n**3. 偶然的机缘：**\n{q3}"
                    history.add_user_message(user_input)
                    st.session_state.exploration_stage = 3
                    st.rerun()

    # 阶段三：觉察“家庭”
    elif stage == 3:
        st.markdown("> **第三阶段：觉察“家庭”(环境影响)**\n> \n> 接下来，我们来探讨需要持续“觉察”的“家庭”与环境影响。")
        with st.form("stage3_form"):
            q1 = st.text_area("1. 你的家庭或重要亲友，对你的职业有什么样的期待？", height=100)
            q2 = st.text_area("2. 有没有哪位榜样（名人、长辈或同辈）对你的职业选择产生了影响？", height=100)
            q3 = st.text_area("3. 你身边的“圈子”（例如朋友、同学）主要从事哪些工作？这对你有什么潜在影响？", height=100)
            submitted = st.form_submit_button("提交关于“家庭”的分析", use_container_width=True)
            if submitted:
                if not q1 or not q2 or not q3:
                    st.warning("请完整填写三个问题的回答。")
                else:
                    user_input = f"### 关于“家庭”的分析\n\n**1. 家庭的期待：**\n{q1}\n\n**2. 榜样的影响：**\n{q2}\n\n**3. 圈子的影响：**\n{q3}"
                    history.add_user_message(user_input)
                    st.session_state.exploration_stage = 4
                    st.rerun()

    # 阶段四：AI 整合报告与最终行动
    elif stage == 4:
        st.markdown("> **第四阶段：AI 智慧整合与行动计划**")
        with st.chat_message("ai", avatar="🤖"):
            full_conversation = "\n\n".join(
                [f"**用户关于 {msg.content.split('###')[1].strip()}**\n{msg.content.split('###')[2].strip()}" for msg in
                 history.messages if isinstance(msg, HumanMessage)])

            stage4_prompt = ChatPromptTemplate.from_template(GLOBAL_PERSONA + """
            作为一名智慧且富有洞察力的职业发展教练，请严格根据以下用户在“我”、“社会”、“家庭”三个阶段的完整回答，为用户生成一份整合分析与建议报告。
            你的任务是 synthesise 用户的输入，并提出具体、可行的建议。报告必须包含以下三个部分，并使用清晰的Markdown格式：

            ### 1. 初步决策方向建议
            - 基于用户的专业、经历、兴趣、以及风险偏好，提出1-2个具体且可行的职业方向。
            - 必须清晰地解释为什么这些方向是合适的，将你的建议与用户之前的回答（例如他的技能、担忧、价值观）联系起来。

            ### 2. 预期“收入”分析
            - 针对你建议的方向，分析其潜在的“收入”。
            - 这不仅包括物质上的“金钱回报”，还必须包括用户看重的非物质“价值回报”（例如：稳定性、成就感、创造性等）。

            ### 3. 第一个“行动”建议
            - 为用户建议一个具体的、低风险的、下周就能完成的第一个行动步骤，用以探索你提出的方向。
            - 这个建议必须非常务实（例如：观看一门具体的公开课、在xx平台上找一位从业者咨询、分析一个相关公司的财报等）。
            ---
            以下是用户的完整回答: 
            {conversation_history}
            ---
            """)
            stage4_chain = stage4_prompt | llm
            with st.spinner("AI教练正在全面分析您的回答，为您生成整合报告..."):
                response_stream = stage4_chain.stream({"conversation_history": full_conversation})
                report_content = st.write_stream(response_stream)
            history.add_ai_message(report_content)

        st.session_state.exploration_stage = 5  # 進入最終提問階段
        st.rerun()

    # 阶段五：最终行动确认
    elif stage == 5:
        final_prompt = "> AI教练已根据您的回答，为您提供了一份整合分析与建议。这份报告是为您量身打造的起点，而非终点。\n>\n> 请仔细阅读报告，然后回答最后一个、也是最重要的问题：\n> **您自己决定要采取的、下周可以完成的第一个具体行动是什么？** (这可以是对AI建议的采纳、修改，或是您自己全新的想法)"
        st.markdown(final_prompt)
        if user_input := st.chat_input("请在此输入您的最终行动计划..."):
            history.add_user_message(user_input)
            st.session_state.exploration_stage = 6
            st.rerun()

    # 阶段六：完成
    elif stage == 6:
        with st.container(border=True):
            st.success("恭喜！您已完成本次探索的全过程。")
            st.info("最终的决策权在您手中，希望这次的探索能为您提供有价值的参考。您可以随时返回主菜单开始新的探索。")


def render_decision_mode(llm):
    """渲染 Offer 决策分析模式的 UI 和逻辑。"""
    st.header("模式二: Offer 决策分析")
    with st.container(border=True):
        st.info("请输入两个Offer的关键信息，AI将为您生成一份结构化的对比分析报告。")
        chain = ChatPromptTemplate.from_template(GLOBAL_PERSONA + """
        You are an expert career advisor. Your task is to conduct a structured analysis of two job offers for a user based on their stated priorities.
        Offer A Details: {offer_a_details}
        Offer B Details: {offer_b_details}
        User Priorities: {user_priorities_sorted_list}
        Please perform the following steps and structure your entire response in clear, easy-to-read markdown:
        1. Create a Comparison Table
        2. Personalized Priority Matching Analysis
        3. Pros and Cons Analysis
        4. Risk Alert & Mitigation
        5. Recommendation and Key Questions
        """) | llm

        st.subheader("第一步：请填写 Offer 的核心信息")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            offer_a = st.text_area("Offer A 关键信息", height=200,
                                   placeholder="例如: 公司名、职位、薪资、地点、优点、顾虑等")
        with col2:
            offer_b = st.text_area("Offer B 关键信息", height=200,
                                   placeholder="同样，包括公司名、职位、薪资、地点、优点、顾虑等")

        st.subheader("第二步：(可选) 添加你的个人偏好")
        priorities_options = ["职业成长", "薪资福利", "工作生活平衡", "团队氛围", "公司稳定性"]
        user_priorities = st.multiselect("请按重要性依次选择你的职业偏好：", options=priorities_options,
                                         help="您选择的第一个选项代表您最看重的因素，以此类推。")

        if st.button("生成对比分析报告", use_container_width=True):
            if not offer_a or not offer_b:
                st.warning("请输入两个Offer的信息后再生成报告。")
            else:
                with st.spinner("正在为您生成Offer分析报告..."):
                    priorities_text = ", ".join(user_priorities) if user_priorities else "用户未指定明确的优先级顺序"
                    response_stream = chain.stream({"offer_a_details": offer_a, "offer_b_details": offer_b,
                                                    "user_priorities_sorted_list": priorities_text})
                    st.markdown("---");
                    st.subheader("📋 Offer对比分析报告");
                    st.write_stream(response_stream)


def render_communication_mode(llm):
    """渲染家庭沟通模拟模式的 UI 和逻辑，增加复盘功能。"""
    st.header("模式三: 家庭沟通模拟")

    if not st.session_state.get('sim_started', False):
        with st.container(border=True):
            st.info("在这里，AI可以扮演您的家人，帮助您练习如何沟通职业规划，并提供复盘建议。")
            my_choice = st.text_input("首先, 请告诉我你想要和家人沟通的职业选择是什么?")
            family_concern = st.text_area("你认为他们主要的担忧会是什么?",
                                          placeholder="例如: 工作不稳定、不是铁饭碗、离家太远等")
            if st.button("开始模拟"):
                if not my_choice or not family_concern:
                    st.warning("请输入您的职业选择和预想的家人担忧。")
                else:
                    st.session_state.my_choice = my_choice
                    st.session_state.family_concern = family_concern
                    st.session_state.sim_started = True
                    st.session_state.debrief_requested = False
                    st.session_state.chat_history['communication_session'] = ChatMessageHistory()
                    initial_ai_prompt = f"孩子，关于你想做“{my_choice}”这个事，我有些担心。我主要是觉得它“{family_concern}”。我们能聊聊吗？"
                    get_session_history("communication_session").add_ai_message(initial_ai_prompt)
                    st.rerun()

    if st.session_state.get('sim_started', False):
        st.success(f"模拟开始！AI正在扮演担忧您选择 “{st.session_state.my_choice}” 的家人。")

        history = get_session_history("communication_session")
        with st.container():
            for msg in history.messages:
                avatar = "🧑‍💻" if isinstance(msg, HumanMessage) else "🧓"
                st.chat_message(msg.type, avatar=avatar).markdown(msg.content)

        if not st.session_state.get('debrief_requested', False):
            communication_prompt = ChatPromptTemplate.from_messages([
                ("system", GLOBAL_PERSONA + f"""
                现在，你将扮演一个关心孩子但思想略显传统的家人。
                - 你的核心担忧是: "{st.session_state.family_concern}"
                - 你的对话目标是：反复确认孩子是否考虑清楚了这些担忧，而不是轻易被说服。
                - 你的语气应该像一个真实的、有自己立场和情绪的家人，可以固执，可以表达失望或不解，但最终的出发点是爱和关心。
                - 不要轻易放弃你的担忧，直到用户给出了非常有说服力、能让你安心的理由。
                """),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            chain_with_history = RunnableWithMessageHistory(
                communication_prompt | llm,
                lambda s: get_session_history(s),
                input_messages_key="input",
                history_messages_key="history"
            )

            if user_input := st.chat_input("你的回应:"):
                with st.spinner("..."):
                    chain_with_history.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": "communication_session"}}
                    )
                st.rerun()

            if len(history.messages) > 2:
                if st.button("结束模拟并获取复盘建议"):
                    st.session_state.debrief_requested = True
                    st.rerun()

        else:
            with st.container(border=True):
                st.info("对话已结束。AI教练正在为您复盘刚才的沟通表现...")
                full_conversation = "\n".join(
                    [f"{'我' if isinstance(msg, HumanMessage) else '家人'}: {msg.content}" for msg in history.messages]
                )
                debrief_prompt = ChatPromptTemplate.from_template(GLOBAL_PERSONA + """
                你现在切换回职业发展教练的角色。以下是一份用户与你扮演的“家人”之间的沟通记录。
                用户的目标是说服家人，让他们理解并减少对自己职业选择({my_choice})的担忧({family_concern})。

                请你从一个专业沟通教练的角度，对用户的表现进行全面、有建设性的复盘。复盘报告必须包括：

                ### 1. 沟通亮点 (做得好的地方)
                - 指出用户在对话中展现出的同理心、清晰的逻辑或有效安抚情绪的具体话语。

                ### 2. 可优化点 (可以更好的地方)
                - 分析用户在哪些地方可能错失了机会，或者哪些回应可能激化了矛盾。
                - 是否有效回应了家人的核心关切点？

                ### 3. 具体话术建议
                - 针对可优化点，提供1-2个具体的、可直接使用的“话术”或“表达方式”建议。例如：“当家人说‘不稳定’时，你可以尝试这样回应：‘我理解您的担心，我也为自己规划了...’”

                请以富有同情心和启发性的方式呈现这份报告。
                ---
                对话记录如下:
                {conversation_history}
                ---
                """)
                debrief_chain = debrief_prompt | llm

                with st.spinner("正在生成沟通复盘报告..."):
                    response_stream = debrief_chain.stream({
                        "my_choice": st.session_state.my_choice,
                        "family_concern": st.session_state.family_concern,
                        "conversation_history": full_conversation
                    })
                    st.subheader("📋 沟通表现复盘报告")
                    st.write_stream(response_stream)


def render_company_info_mode(llm):
    """渲染企业信息速览模式的 UI 和逻辑。"""
    st.header("模式四: 企业信息速览")
    with st.container(border=True):
        st.info("请输入公司全名，AI将模拟网络抓取并为您生成一份核心信息速览报告。")
        chain = ChatPromptTemplate.from_template(GLOBAL_PERSONA + """
        You are a professional business analyst AI. Your task is to provide a concise and structured overview of a given company, as if you had scraped its public information from the web. The report should be well-organized for a job seeker.

        Company Name: {company_name}

        Please structure your report in markdown with the following sections:

        ### 1. 公司简介 (Company Overview)
        - Core business, main products/services, and target market.

        ### 2. 近期动态与新闻 (Recent Developments & News)
        - Summarize 1-2 key recent events, such as major product launches, financial results, or strategic partnerships.

        ### 3. 企业文化与价值观 (Culture & Values)
        - Briefly describe the stated or perceived company culture.

        ### 4. 热门招聘方向 (Hiring Trends)
        - Based on publicly available information, what types of roles or departments seem to be in high demand at the company? (e.g., "Engineering, particularly in AI/ML", "Sales and Business Development").

        Your tone should be professional, objective, and informative.
        """) | llm
        company_name = st.text_input("请输入公司名称:", placeholder="例如：阿里巴巴、腾讯、字节跳动")
        if st.button("生成速览报告", use_container_width=True):
            if not company_name:
                st.warning("请输入公司名称。")
            else:
                with st.spinner(f"正在生成关于“{company_name}”的信息报告..."):
                    response_stream = chain.stream({"company_name": company_name})
                    st.markdown("---");
                    st.subheader(f"📄 {company_name} - 核心信息速览");
                    st.write_stream(response_stream)


def render_panoramic_mode(llm):
    """渲染职业路径全景规划模式。"""
    st.header("模式五: 职业路径全景规划")
    history = get_session_history("panoramic_session")
    stage = st.session_state.get('panoramic_stage', 1)

    # 渲染历史消息
    for msg in history.messages:
        avatar = "🧑‍💻" if isinstance(msg, HumanMessage) else "🤖"
        with st.chat_message(msg.type, avatar=avatar):
            st.markdown(msg.content, unsafe_allow_html=True)

    meta_prompt_template = GLOBAL_PERSONA + """
    You are an expert career strategist, guiding the user through a panoramic career path analysis based on the "Specialty -> Profession -> Enterprise -> Industry Chain -> Industry" framework. You are currently in Stage {current_stage}.

    User's Core Competency Profile: {user_profile}
    User's Chosen Profession(s): {chosen_professions}
    User's Chosen Company Type: {chosen_company_type}

    Your Task is to execute the current stage's logic.

    --- STAGE-SPECIFIC INSTRUCTIONS ---

    **Stage 1 (Core Competency Assessment):**
    The user has just provided their self-assessment. Your task is to briefly summarize their core competency profile in an encouraging tone to confirm understanding.
    *Edge Case Handling:* If the user's input is too vague or contradictory (e.g., skills and motivation mismatch), your first task is to ask a clarifying question instead of summarizing. For example: "我注意到您的技能偏向技术分析，但动机更侧重于人际沟通。您能否分享一个将两者结合的经历？这有助于我更精准地为您匹配职业。"

    **Stage 2 (Profession Concretization):**
    Based on the user's profile, brainstorm and present 3-5 concrete professions (职业). For each, explain its connection to the user's skills and motivations in one sentence.
    *Format:* Use a markdown list. Example: "- **数据分析师**: 能很好地结合您在 `数据处理` 上的技能和 `通过数据洞察趋势` 的动机。"
    Finally, prompt the user to select 1-2 professions they are most interested in.

    **Stage 3 (Representative Enterprise Targeting):**
    Based on the user's chosen profession, identify 3-5 representative companies (企业). Include a mix of stable, established companies and innovative, high-growth ones. Provide a one-sentence description for each.
    *Edge Case Handling:* If the field is too new or niche for clear representative companies, state it honestly. Example: "这是一个非常前沿和令人兴奋的领域！这意味着路径尚未定型，充满了机会与挑战。让我们专注于构建您的可迁移核心能力，并找出这个领域的开创性组织。"
    Finally, ask the user which type of company they lean towards (e.g., "安定型" or "成长型").

    **Stage 4 (Industry Chain Position Analysis):**
    Based on the chosen profession and company type, explain the concept of the industry chain (产业链) in simple terms. Describe where the target role/company typically fits. Mention related upstream/downstream sectors.

    **Stage 5 (Industry Trend & Transformation Insight):**
    Provide a concise analysis of the broader industry (行业). Summarize 2-3 key trends and 1-2 potential future transformations. Connect these back to the user's potential role.
    *Edge Case Handling:* If the industry is declining, handle it sensitively. Present the data and then shift focus to adjacent, growing industries where the user's core competencies are valuable. Example: "数据显示传统的 [行业X] 正面临转型挑战，但您所具备的 [技能A] 和 [技能B] 在高速发展的 [行业Y] 领域正变得炙手可热。我们不妨探讨一下，如何将您的核心能力迁移到这个新赛道。"

    **Stage 6 (Summary & Strategic Planning):**
    Generate a full markdown report summarizing findings from stages 2-5. Conclude with a "战略展望 (Strategic Outlook)" section. Pose 2-3 forward-looking questions to help the user plan their next steps.
    """

    chain = ChatPromptTemplate.from_template(meta_prompt_template) | llm

    if stage == 1 and st.session_state.user_profile is None:
        st.markdown(
            "> 你好！我是你的职业路径规划助手。想不想像打开一张地图一样，清晰地看到你的个人能力如何一步步通往一个具体的行业，并看清未来的发展趋势？\n> 这个“职业路径全景规划”模式，将引导你从自我认知出发，连接职业、企业、产业链，最终洞察整个行业的未来。让我们开始第一步，也是最重要的一步：**认识你自己**。")
        with st.form("profile_form"):
            st.subheader("请根据以下五个维度，描述你的“核心能力”：")
            edu = st.text_area("学历背景 (Education)", placeholder="你的专业、学位、以及相关的核心课程")
            skills = st.text_area("核心技能 (Skills)",
                                  placeholder="你最擅长的3-5项硬技能或软技能，如编程、产品设计、沟通、数据分析等")
            exp = st.text_area("相关经验 (Experience)", placeholder="相关的实习、工作项目、或个人作品集")
            char = st.text_area("品行特质 (Character)",
                                placeholder="你认为自己最重要的职业品行或工作风格，如严谨、创新、有责任心")
            motiv = st.text_area("内在动机 (Motivation)", placeholder="在工作中，什么最能给你带来成就感和满足感？")

            submitted = st.form_submit_button("提交我的能力画像", use_container_width=True)
            if submitted:
                if not all([edu, skills, exp, char, motiv]):
                    st.warning("请填写所有五个维度的信息，以便进行准确分析。")
                else:
                    profile_text = f"学历背景: {edu}\n核心技能: {skills}\n相关经验: {exp}\n品行特质: {char}\n内在动机: {motiv}"
                    st.session_state.user_profile = profile_text
                    history.add_user_message(f"这是我的能力画像：\n{profile_text}")
                    st.session_state.panoramic_stage = 2
                    st.rerun()

    elif st.session_state.user_profile is not None and stage < 7:
        if f"stage_{stage}_started" not in st.session_state:
            with st.chat_message("ai", avatar="🤖"):
                with st.spinner("AI 正在为您分析下一步..."):
                    response_stream = chain.stream({
                        "current_stage": stage,
                        "user_profile": st.session_state.user_profile,
                        "chosen_professions": st.session_state.get('chosen_professions', 'N/A'),
                        "chosen_company_type": st.session_state.get('chosen_company_type', 'N/A')
                    })
                    response_content = st.write_stream(response_stream)
                    history.add_ai_message(response_content)
            st.session_state[f"stage_{stage}_started"] = True
            st.rerun()

        if stage < 6:
            if user_input := st.chat_input("请在此输入您的选择或想法..."):
                history.add_user_message(user_input)
                if stage == 2:
                    st.session_state.chosen_professions = user_input
                elif stage == 3:
                    st.session_state.chosen_company_type = user_input
                st.session_state.panoramic_stage = stage + 1
                st.rerun()
        else:
            with st.container(border=True):
                st.success("恭喜！您已完成本次职业路径全景规划。")
                clarity_score = st.radio(
                    "**本次分析在多大程度上提升了您对个人职业路径的清晰度？** (5分代表非常清晰)",
                    options=[1, 2, 3, 4, 5], index=None, horizontal=True)
                if clarity_score:
                    st.info(f"感谢您的评分！希望这次规划对您有帮助。")


# --- 主应用逻辑 ---
def main():
    """主函数，运行 Streamlit 应用。"""
    llm = get_llm_instance()
    if not llm:
        st.error("无法初始化语言模型，应用程序无法启动。请检查您的 API Key 设置。")
        st.stop()

    with st.sidebar:
        # st.image("https://s2.loli.net/2024/05/31/vCSO5WwR6r2zMGU.png", width=150) # <-- 已将此行注释掉，解决404问题
        if st.session_state.current_mode != "menu":
            if st.button("↩️ 返回主菜单"):
                for key in list(st.session_state.keys()):
                    if key != 'current_mode':
                        del st.session_state[key]
                st.session_state.current_mode = "menu"
                st.rerun()
        st.markdown("---")
        st.caption("© 2025 智慧职业辅导 V5.2")

    modes = {
        "menu": render_menu,
        "exploration": render_exploration_mode,
        "decision": render_decision_mode,
        "communication": render_communication_mode,
        "company_info": render_company_info_mode,
        "panoramic": render_panoramic_mode,
    }

    mode_func = modes.get(st.session_state.current_mode, render_menu)

    if st.session_state.current_mode == "menu":
        mode_func()
    else:
        mode_func(llm)


if __name__ == "__main__":
    main()