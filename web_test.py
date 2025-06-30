import os
import re
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_mermaid import st_mermaid
from PIL import Image
import platform

# --- 頁面設定 (必須是第一個 Streamlit 命令) ---
st.set_page_config(
    page_title="智慧化职业发展辅导系统",
    page_icon="✨",
    layout="wide"
)

# --- UI 美化 CSS 樣式 ---
st.markdown("""
<style>
    /* 明確匯入所需的文字字型和圖示字型 */
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');

    html, body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Helvetica Neue", "PingFang SC", "Microsoft YaHei", sans-serif;
        line-height: 1.65;
        background-color: #F8F9FA;
        color: #495057;
    }

    h1, h2, h3, h4, h5, h6 { color: #212529; font-weight: 700; }
    h1 { font-size: 32px; }
    h2 { font-size: 28px; border-bottom: 2px solid #E9ECEF; padding-bottom: 0.4em; }
    h3 { font-size: 22px; }

    .st-emotion-cache-z5fcl4 {
        display: flex;
        flex-direction: column;
        height: 100%;
    }

    .st-emotion-cache-z5fcl4 .stButton {
        margin-top: auto;
    }

    .st-emotion-cache-z5fcl4 {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 28px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #E9ECEF;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .st-emotion-cache-z5fcl4:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.08);
    }

    .stButton>button {
        border-radius: 8px; border: none; color: white; font-weight: 500;
        padding: 12px 24px; background-image: linear-gradient(135deg, #5D9CEC 0%, #4A90E2 100%);
        transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(74, 144, 226, 0.2);
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(74, 144, 226, 0.3); }

    .stChatMessage { border-radius: 12px; border: 1px solid #E9ECEF; background-color: #FFFFFF; padding: 16px; margin-bottom: 1rem; }
    .st-emotion-cache-T21nqy { background-color: #e3eeff; border-color: #a4c7ff; }

    .stChatInputContainer {
        position: sticky; bottom: 0; background-color: #FFFFFF;
        padding: 12px 0px; border-top: 1px solid #E9ECEF;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- OCR路徑設定 (根據您的實際安裝路徑修改) ---
try:
    import pytesseract

    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract at UB Mannheim\tesseract.exe'
except ImportError:
    pass

# --- 初始化 ---
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# --- 全域系統角色 (簡體中文) ---
GLOBAL_PERSONA = "核心角色: 你是一位智慧、专业且富有同理心的职业发展教练与战略规划师。\n语言要求: 你的所有回答都必须使用简体中文。"


# --- LLM 初始化 ---
@st.cache_resource
def get_llm_instance():
    api_key = None;
    key_name = "VOLCENGINE_API_KEY"
    try:
        api_key = st.secrets[key_name]
    except (KeyError, FileNotFoundError):
        api_key = os.getenv(key_name)
    if not api_key:
        st.error(f"错误：未找到 {key_name}。请在 Streamlit Cloud Secrets 或本地 .env 文件中设置它。");
        return None
    try:
        llm = ChatOpenAI(model="deepseek-r1-250528", temperature=0.7, api_key=api_key,
                         base_url="https://ark.cn-beijing.volces.com/api/v3")
        llm.invoke("Hello");
        return llm
    except Exception as e:
        st.error(f"初始化模型时出错: {e}");
        return None


# --- 會話狀態管理 ---
def init_session_state():
    defaults = {"current_mode": "menu", "chat_history": {}, "exploration_stage": 1, "sim_started": False,
                "debrief_requested": False, "panoramic_stage": 1, "user_profile": None, "chosen_professions": None,
                "chosen_region": None, "curriculum_stage": 1, "curriculum_content": None, "chosen_career": None,
                "key_courses_identified": None}
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value


init_session_state()


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in st.session_state.chat_history: st.session_state.chat_history[session_id] = ChatMessageHistory()
    return st.session_state.chat_history[session_id]


# --- UI 渲染函式 ---
def render_menu():
    st.title("✨ 智慧化职业发展辅导系统")
    st.markdown("---")
    st.subheader("欢迎使用！请选择一项功能开始探索：")
    st.write("")
    modes_config = [
        ("exploration", ":compass: 职业目标探索", "通过“我-社会-家庭”框架，系统性地探索内在动机与外在机会。"),
        ("panoramic", ":globe_with_meridians: 职业路径全景规划",
         "从您的核心能力出发，连接职业、企业、地区与产业链，生成您的个人发展蓝图。"),
        ("decision", ":balance_scale: Offer 决策分析", "结构化对比多个Offer，获得清晰的决策建议。"),
        ("company_info", ":office: 企业信息速览", "快速了解目标公司的核心业务、近期动态与热招方向。"),
        ("communication", ":family: 家庭沟通模拟", "与AI扮演的家人进行对话，安全地练习如何表达您的职业选择。"),
        ("curriculum_analysis", ":school: 专业培养方案解析",
         "上传您的专业培养方案，AI将为您解析课程体系，并规划重点学习路径。")
    ]
    cols = st.columns(3)
    for i, (mode_key, title, caption) in enumerate(modes_config):
        with cols[i % 3]:
            with st.container(border=True, height=230):
                st.subheader(title)
                st.caption(caption)
                button_label = f"开始{title.split(' ')[1][:2]}"
                if st.button(button_label, use_container_width=True, key=f"menu_{mode_key}"):
                    st.session_state.current_mode = mode_key
                    init_session_state()
                    st.session_state.current_mode = mode_key
                    st.rerun()
            st.write("")


# ----------------------------------------------------------------
# --- 模式一至五 (完整程式碼) ---
# ----------------------------------------------------------------
def render_exploration_mode(llm):
    st.header("模式一: 职业目标探索")
    history = get_session_history("exploration_session")
    stage = st.session_state.get('exploration_stage', 1)

    def generate_interim_response(user_input, prompt_template):
        with st.chat_message("ai", avatar="🤖"):
            chain = ChatPromptTemplate.from_template(prompt_template) | llm
            with st.spinner("AI教练正在思考..."):
                response_content = st.write_stream(chain.stream({"user_input": user_input}))
            history.add_ai_message(response_content)
        st.session_state.exploration_stage += 1
        st.rerun()

    for msg in history.messages:
        avatar = "🧑‍💻" if isinstance(msg, HumanMessage) else "🤖"
        st.chat_message(msg.type, avatar=avatar).markdown(msg.content, unsafe_allow_html=True)
    if stage == 1:
        with st.chat_message("ai", avatar="🤖"):
            if len(history.messages) == 0:
                welcome_msg = "你好！我将引导你使用“职业目标缘起分析框架”，从“我”、“社会”、“家庭”三个核心维度，系统性地探索你的职业方向。\n\n首先，我们来分析“我”这个核心。请在下方回答："
                st.markdown(welcome_msg)
                history.add_ai_message(welcome_msg)
        questions = ["1. 你的专业是什么？你对它的看法如何？", "2. 你的学校或过往经历，为你提供了怎样的平台与基础？"]
        with st.form("stage1_form"):
            st.markdown("> **第一阶段：分析“我”(可控因素)**")
            responses = [st.text_area(q, height=100, key=f"s1_q{i}") for i, q in enumerate(questions)]
            if st.form_submit_button("提交关于“我”的分析", use_container_width=True):
                if all(responses):
                    input_text = "### 关于“我”的回答\n\n" + "\n\n".join(
                        [f"**{q}**\n{r}" for q, r in zip(questions, responses)])
                    history.add_user_message(input_text)
                    st.session_state.exploration_stage += 1
                    st.rerun()
                else:
                    st.warning("请完整填写所有问题的回答。")
    elif stage in [2, 4, 6]:
        last_user_message = history.messages[-1].content
        prompts = {
            2: GLOBAL_PERSONA + "任务：作为职业教练，对用户刚才提供的关于“我”的信息，给予一段简短、积极的总结和肯定。然后，自然地引出我们下一个要探讨的“社会”维度。\n要求：语言要富有同理心，充满鼓励，不要超过100字。结尾必须是引出下一阶段的提问。\n用户的输入：{user_input}\n你的回应：",
            4: GLOBAL_PERSONA + "任务：作为职业教练，对用户刚才提供的关于“社会”趋势的观察，给予一段简短、富有洞察力的总结。然后，自然地引出我们需要探讨的最后一个维度“家庭”。\n要求：肯定用户观察的价值，语言精炼，不要超过100字。结尾必须是引出下一阶段的提问。\n用户的输入：{user_input}\n你的回应：",
            6: GLOBAL_PERSONA + "任务：作为职业教练，对用户刚才提供的关于“家庭”与环境影响的描述，给予一段富有同理心和理解的回应。然后告诉用户，现在信息已经收集完毕，你将为他整合所有信息并生成最终的分析报告。\n要求：表达理解和共情，语言温暖，不要超过100字。明确告知用户下一步是生成总报告。\n用户的输入：{user_input}\n你的回应："
        }
        generate_interim_response(last_user_message, prompts[stage])
    elif stage in [3, 5]:
        forms = {
            3: ("stage3_form", "> **第二阶段：分析“社会”(外部机会)**", "提交关于“社会”的分析",
                ["1. 你观察到当下有哪些你感兴趣的社会或科技趋势？（例如：AI、大健康、可持续发展等）",
                 "2. 根据你的观察，这些趋势可能带来哪些新的行业或职位机会？",
                 "3. 在你过往的经历中，有没有一些偶然的机缘或打工经验，让你对某个领域产生了特别的了解？"],
                "### 关于“社会”的回答"),
            5: ("stage5_form", "> **第三阶段：觉察“家庭”(环境影响)**", "提交关于“家庭”的分析",
                ["1. 你的家庭或重要亲友，对你的职业有什么样的期待？", "2. 有没有哪位榜样对你的职业选择产生了影响？",
                 "3. 你身边的“圈子”（例如朋友、同学）主要从事哪些工作？这对你有什么潜在影响？"],
                "### 关于“家庭”的回答")
        }
        form_key, title, button_text, questions, header = forms[stage]
        with st.form(form_key):
            st.markdown(title)
            responses = [st.text_area(q, height=100, key=f"s{stage}_q{i}") for i, q in enumerate(questions)]
            if st.form_submit_button(button_text, use_container_width=True):
                if all(responses):
                    input_text = f"{header}\n\n" + "\n\n".join([f"**{q}**\n{r}" for q, r in zip(questions, responses)])
                    history.add_user_message(input_text)
                    st.session_state.exploration_stage += 1
                    st.rerun()
                else:
                    st.warning("请完整填写所有问题的回答。")
    elif stage == 7:
        st.markdown("> **第四阶段：AI 智慧整合与行动计划**")
        with st.chat_message("ai", avatar="🤖"):
            full_conversation = "\n\n".join([msg.content for msg in history.messages if isinstance(msg, HumanMessage)])
            stage4_prompt = ChatPromptTemplate.from_template(
                GLOBAL_PERSONA + "作为一名智慧且富有洞察力的职业发展教练，请严格根据以下用户在“我”、“社会”、“家庭”三个阶段的完整回答，为用户生成一份结构清晰、富有洞见的整合分析与建议报告。报告必须包含以下三个核心部分：\n\n**1. 核心洞察总结：**\n   - **优势与机遇 (S&O):** 结合用户的“我”和“社会”，提炼出 2-3 个最关键的优势与外部机遇的结合点。\n   - **挑战与关注 (C&A):** 结合用户的“我”的潜在局限和“家庭/环境”的影响，指出 1-2 个需要特别关注和应对的挑战。\n\n**2. 职业方向建议 (探索象限):**\n   - 基于以上分析，提出 2-3 个具体的、可探索的职业方向建议。\n   - 对每个方向，用一句话点明它为什么与用户的“我-社会-家庭”分析相匹配。\n\n**3. 下一步行动清单 (Action Plan):**\n   - 提供一个包含 3-5 个具体、可执行的“轻量级”行动建议。\n\n**报告风格要求：**\n- 语言专业、积极、富有启发性，但也要实事求是。\n- 使用 Markdown 格式，条理清晰，重点突出。\n- 直接输出报告内容，无需重复用户的回答。\n\n---\n以下是用户的完整回答:\n{conversation_history}\n---")
            stage4_chain = stage4_prompt | llm
            with st.spinner("AI教练正在全面分析您的回答，生成最终报告..."):
                response_content = st.write_stream(stage4_chain.stream({"conversation_history": full_conversation}))
            history.add_ai_message(response_content)
        st.session_state.exploration_stage += 1
        st.rerun()
    elif stage == 8:
        st.markdown(
            "> AI教练已根据您的回答，为您提供了一份整合分析与建议。这份报告是为您量身打造的起点，而非终点。\n>\n> 请仔细阅读报告，然后回答最后一个、也是最重要的问题：\n> **您自己决定要采取的、下周可以完成的第一个具体行动是什么？**")
        if user_input := st.chat_input("请在此输入您的最终行动计划..."):
            history.add_user_message(f"我的最终行动计划是：{user_input}")
            st.session_state.exploration_stage += 1
            st.rerun()
    elif stage == 9:
        with st.chat_message("ai", avatar="🤖"):
            final_msg = "太棒了！明确的行动是推动一切改变的开始。预祝你行动顺利，在职业探索的道路上不断有新的发现和收获！"
            st.markdown(final_msg)
            history.add_ai_message(final_msg)
        st.success("恭喜！您已完成本次探索的全过程。")
        st.session_state.exploration_stage += 1


def render_decision_mode(llm):
    st.header("模式二: Offer 决策分析")
    with st.container(border=True):
        st.info("请输入两个Offer的关键信息，AI将为您生成一份结构化的对比分析报告。")
        chain = ChatPromptTemplate.from_template(
            GLOBAL_PERSONA + "作为一名专业的职业顾问，你的任务是帮助用户对比两个Offer，并根据他们提供的个人偏好，生成一份结构化、逻辑清晰的分析报告。\n\n**输入信息:**\n- **Offer A 详情:** {offer_a_details}\n- **Offer B 详情:** {offer_b_details}\n- **用户的个人偏好 (按重要性排序):** {user_priorities_sorted_list}\n\n**输出报告要求:**\n1.  **开篇总结:** 首先，对两个Offer的核心亮点进行一句话总结。\n2.  **多维度对比分析:**\n    -   根据用户选择的偏好维度进行逐一对比。\n    -   如果用户未提供偏好，则使用默认的通用维度（如：薪酬、发展、稳定性、通勤、文化）进行分析。\n    -   在每个维度下，清晰地列出Offer A和Offer B各自的表现，并给出一个简短的小结。\n    -   使用Markdown的表格或项目符号，让对比一目了然。\n3.  **综合建议:**\n    -   基于前面的多维度分析，给出一个综合性的决策建议。\n    -   明确指出哪个Offer与用户的偏好更匹配，并解释原因。\n4.  **风格要求:** 语言客观、中立、富有逻辑，避免使用绝对化的词语。") | llm
        st.subheader("第一步：请填写 Offer 的核心信息")
        col1, col2 = st.columns(2, gap="large");
        with col1:
            offer_a = st.text_area("Offer A 关键信息", height=200,
                                   placeholder="例如：\n公司: A科技\n职位: 初级产品经理\n薪资: 15k * 14薪\n地点: 上海张江...")
        with col2:
            offer_b = st.text_area("Offer B 关键信息", height=200,
                                   placeholder="例如：\n公司: B集团\n职位: 管培生\n薪资: 13k * 16薪 + 2w签字费\n地点: 北京海淀...")
        st.subheader("第二步：(可选) 添加你的个人偏好")
        priorities_options = ["职业成长", "薪资福利", "工作生活平衡", "团队氛围", "公司稳定性"]
        user_priorities = st.multiselect("请按重要性依次选择你的职业偏好：", options=priorities_options)
        if st.button("生成对比分析报告", use_container_width=True):
            if not offer_a or not offer_b:
                st.warning("请输入两个Offer的信息。")
            else:
                with st.spinner("正在为您生成Offer分析报告..."):
                    priorities_text = ", ".join(user_priorities) if user_priorities else "用户未指定"
                    response_stream = chain.stream({"offer_a_details": offer_a, "offer_b_details": offer_b,
                                                    "user_priorities_sorted_list": priorities_text})
                    st.markdown("---");
                    st.subheader("📋 Offer对比分析报告");
                    st.write_stream(response_stream)


def render_communication_mode(llm):
    st.header("模式三: 家庭沟通模拟")
    if not st.session_state.get('sim_started', False):
        with st.container(border=True):
            st.info("在这里，AI可以扮演您的家人，帮助您练习如何沟通职业规划，并提供复盘建议。")
            my_choice = st.text_input("你想和家人沟通的职业选择是？")
            family_concern = st.text_area("你认为他们主要的担忧会是什么？",
                                          placeholder="例如: 工作不稳定、不是铁饭碗、离家太远等")
            if st.button("开始模拟"):
                if not my_choice or not family_concern:
                    st.warning("请输入您的职业选择和预想的家人担忧。")
                else:
                    st.session_state.my_choice = my_choice;
                    st.session_state.family_concern = family_concern
                    st.session_state.sim_started = True;
                    st.session_state.debrief_requested = False
                    initial_ai_prompt = f"孩子，关于你想做“{my_choice}”这个事，我有些担心。我主要是觉得它“{family_concern}”。我们能聊聊吗？"
                    get_session_history("communication_session").add_ai_message(initial_ai_prompt);
                    st.rerun()
    if st.session_state.get('sim_started', False):
        st.success(f"模拟开始！AI正在扮演担忧您选择 “{st.session_state.my_choice}” 的家人。")
        history = get_session_history("communication_session")
        with st.container():
            for msg in history.messages:
                avatar = "🧑‍💻" if isinstance(msg, HumanMessage) else "🧓";
                st.chat_message(msg.type, avatar=avatar).markdown(msg.content)
        if not st.session_state.get('debrief_requested', False):
            communication_prompt = ChatPromptTemplate.from_messages([("system",
                                                                      GLOBAL_PERSONA + f"现在，你将扮演一个关心孩子但思想略显传统的家人（父亲/母亲）。\n你的背景：你非常爱自己的孩子，但对新兴职业不太了解，更看重稳定、体面的工作。\n你的任务：\n1. 你的开场白已经由系统给出。\n2. 在接下来的对话中，持续表达你对孩子职业选择({st.session_state.my_choice})的担忧({st.session_state.family_concern})。\n3. 你的语气要真诚、关切，可以略带固执，但最终目的是希望孩子能过得好。\n4. 根据用户的回应进行追问。\n5. 保持你的角色，直到用户点击“结束模拟”。"),
                                                                     MessagesPlaceholder(variable_name="history"),
                                                                     ("human", "{input}")])
            chain_with_history = RunnableWithMessageHistory(communication_prompt | llm,
                                                            lambda s: get_session_history(s),
                                                            input_messages_key="input", history_messages_key="history")
            if user_input := st.chat_input("你的回应:"):
                with st.spinner("..."): chain_with_history.invoke({"input": user_input}, config={
                    "configurable": {"session_id": "communication_session"}}); st.rerun()
            if len(history.messages) > 2:
                if st.button("结束模拟并获取复盘建议"): st.session_state.debrief_requested = True; st.rerun()
        else:
            with st.container(border=True):
                st.info("对话已结束。AI教练正在为您复盘刚才的沟通表现...")
                full_conversation = "\n".join(
                    [f"{'我' if isinstance(msg, HumanMessage) else '家人'}: {msg.content}" for msg in history.messages])
                debrief_prompt = ChatPromptTemplate.from_template(
                    GLOBAL_PERSONA + "你现在切换回职业发展教练的角色。\n任务：请对以下这段“我”与“家人”关于职业选择的沟通对话进行复盘，并生成一份结构化的沟通表现报告。\n\n**已知背景:**\n- 我的职业选择: {my_choice}\n- 家人预设的担忧: {family_concern}\n\n**沟通记录:**\n{conversation_history}\n\n**复盘报告要求:**\n1.  **沟通亮点 (做得好的地方):**\n    -   识别并表扬我在对话中使用的有效沟通技巧。\n2.  **可提升点 (可以做得更好的地方):**\n    -   建设性地指出沟通中可以改进的地方。\n3.  **核心策略建议:**\n    -   提供 2-3条具体的、可操作的沟通策略。\n\n报告风格需专业、客观、富有建设性。")
                debrief_chain = debrief_prompt | llm
                with st.spinner("正在生成沟通复盘报告..."):
                    response_stream = debrief_chain.stream(
                        {"my_choice": st.session_state.my_choice, "family_concern": st.session_state.family_concern,
                         "conversation_history": full_conversation})
                    st.subheader("📋 沟通表现复盘报告");
                    st.write_stream(response_stream)


def render_company_info_mode(llm):
    st.header("模式四: 企业信息速览")
    with st.container(border=True):
        st.info("请输入公司全名，AI将为您综合网络信息，生成一份核心信息速览报告。")
        chain = ChatPromptTemplate.from_template(
            GLOBAL_PERSONA + "你是一位专业的商业分析师AI。\n任务：请为用户查询并生成一份关于 **{company_name}** 的核心信息速览报告。\n\n**报告必须包含以下部分:**\n1.  **一句话总结:** 用一句话精准概括该公司的核心业务和市场地位。\n2.  **公司简介:** 简要介绍公司的成立背景、主营业务、关键产品或服务。\n3.  **近期动态与新闻:**\n    -   总结 1-2 条该公司近期的重要动态、战略调整或相关的行业新闻。\n4.  **热招方向分析:**\n    -   分析该公司近期的招聘趋势，指出 2-3 个重点招聘的职能方向或岗位类型。\n5.  **SWOT分析 (简版):**\n    -   **优势(S):** 最主要的竞争优势是什么？\n    -   **劣势(W):** 面临的主要挑战或不足是什么？\n    -   **机会(O):** 外部环境带来了哪些发展机会？\n    -   **威胁(T):** 市场或竞争带来了哪些潜在威胁？\n\n请确保报告内容客观、信息凝练、条理清晰。") | llm
        company_name = st.text_input("请输入公司名称:", placeholder="例如：阿里巴巴、腾讯、字节跳动")
        if st.button("生成速览报告", use_container_width=True):
            if not company_name:
                st.warning("请输入公司名称。")
            else:
                with st.spinner(f"正在为您分析“{company_name}”..."):
                    response_stream = chain.stream({"company_name": company_name})
                    st.markdown("---");
                    st.subheader(f"📄 {company_name} - 核心信息速览");
                    st.write_stream(response_stream)


def render_panoramic_mode(llm):
    st.header("模式五: 职业路径全景规划")
    history = get_session_history("panoramic_session")
    stage = st.session_state.get('panoramic_stage', 1)
    for msg in history.messages:
        avatar = "🧑‍💻" if isinstance(msg, HumanMessage) else "🤖"
        with st.chat_message(msg.type, avatar=avatar):
            if msg.type == 'ai' and "```mermaid" in msg.content:
                parts = msg.content.split("```mermaid");
                st.markdown(parts[0])
                mermaid_section = "```mermaid" + parts[1];
                mermaid_match = re.search("```mermaid\n(.*?)\n```", mermaid_section, re.DOTALL)
                if mermaid_match:
                    mermaid_code = mermaid_match.group(1);
                    st.subheader("产业链可视化图表")
                    with st.container(border=True): st_mermaid(mermaid_code.strip())
                after_diagram_content = mermaid_section.split("```")[-1]
                if after_diagram_content.strip(): st.markdown(after_diagram_content)
            else:
                st.markdown(msg.content)
    meta_prompt_template = GLOBAL_PERSONA + "You are an expert career strategist, guiding the user through a multi-stage panoramic career path analysis. You are currently in Stage {current_stage}.\nUser's Core Competency Profile: {user_profile}\nUser's Chosen Profession(s): {chosen_professions}\nUser's Chosen Region(s): {chosen_region}\n\nYour Task is to execute the current stage's logic.\n--- STAGE-SPECIFIC INSTRUCTIONS ---\n**Stage 1:** Do not respond.\n**Stage 2 (Profession Concretization):** Based on the user's profile, present 3-5 concrete professions and prompt the user to select one or two.\n**Stage 3 (Enterprise & Region Targeting):** Based on the chosen profession, identify representative companies and primary geographic clusters in China. Prompt the user for their geographical preference.\n**Stage 4 (Final Comprehensive Report):** The user has provided all inputs. Generate a single, comprehensive report with the following sections:\n    1.  **产业链位置分析:** Explain the role's position in the industry chain. Then, generate a Mermaid flowchart (`graph TD`). **CRITICAL SYNTAX RULE:** To create a line break inside a node's text, you MUST use the `<br>` HTML tag, and the entire text MUST be enclosed in double quotes.\n    2.  **行业趋势与“365理论”定性:** Analyze industry trends and classify the industry as '战略型', '支柱型', or '趋势型'.\n    3.  **目标职能要求与差距分析:** List typical requirements and perform a gap analysis.\n    4.  **个人发展蓝图:** Provide 2-3 actionable suggestions.\n    5.  **总结与战略规划:** Provide a concluding summary.\n    6.  **【CRITICAL】战略性思考点:** Finally, conclude with this section, providing 2-3 introspective questions for the user's long-term reflection. **DO NOT ask the user to answer them now.**"
    chain = ChatPromptTemplate.from_template(meta_prompt_template) | llm
    if stage == 1:
        st.markdown("> 你好！我是你的职业路径规划助手。让我们从认识你自己开始。")
        with st.form("profile_form"):
            st.subheader("请根据以下五个维度，描述你的“核心能力”：");
            edu = st.text_area("学历背景", placeholder="你的专业、学位、以及相关的核心课程");
            skills = st.text_area("核心技能", placeholder="你最擅长的3-5项硬技能或软技能");
            exp = st.text_area("相关经验", placeholder="相关的实习、工作项目、或个人作品集");
            char = st.text_area("品行特质", placeholder="你认为自己最重要的职业品行或工作风格");
            motiv = st.text_area("内在动机", placeholder="在工作中，什么最能给你带来成就感？")
            if st.form_submit_button("提交我的能力画像", use_container_width=True):
                if all([edu, skills, exp, char, motiv]):
                    profile_text = f"学历背景: {edu}\n核心技能: {skills}\n相关经验: {exp}\n品行特质: {char}\n内在动机: {motiv}"
                    st.session_state.user_profile = profile_text;
                    history.add_user_message(f"这是我的能力画像：\n{profile_text}")
                    st.session_state.panoramic_stage = 2;
                    st.rerun()
                else:
                    st.warning("请填写所有五个维度的信息。")
    elif stage in [2, 3]:
        if len(history.messages) % 2 != 0:
            with st.chat_message("ai", avatar="🤖"):
                with st.spinner("AI 正在为您分析..."):
                    response_stream = chain.stream(
                        {"current_stage": stage, "user_profile": st.session_state.user_profile,
                         "chosen_professions": st.session_state.get('chosen_professions', 'N/A'),
                         "chosen_region": st.session_state.get('chosen_region', 'N/A')})
                    response_content = st.write_stream(response_stream);
                    history.add_ai_message(response_content)
            st.rerun()
        st.info("👇 请在下方的输入框中输入您的选择或想法...", icon="💡")
        if user_input := st.chat_input("请在此输入您的选择或想法..."):
            history.add_user_message(user_input)
            if stage == 2:
                st.session_state.chosen_professions = user_input
            elif stage == 3:
                st.session_state.chosen_region = user_input
            st.session_state.panoramic_stage += 1;
            st.rerun()
    elif stage == 4:
        if len(history.messages) % 2 != 0:
            with st.chat_message("ai", avatar="🤖"):
                st.markdown("好的，已收到您的所有信息。现在，我将为您生成一份完整的综合分析报告...")
                with st.spinner("AI 正在为您生成最终报告..."):
                    response_stream = chain.stream({"current_stage": 4, "user_profile": st.session_state.user_profile,
                                                    "chosen_professions": st.session_state.get('chosen_professions',
                                                                                               'N/A'),
                                                    "chosen_region": st.session_state.get('chosen_region', 'N/A')})
                    response_content = st.write_stream(response_stream);
                    history.add_ai_message(response_content)
            st.session_state.panoramic_stage += 1;
            st.rerun()
    elif stage == 5:
        st.success("恭喜！您已完成本次职业路径全景规划。")
        st.info("您可以向上滚动查看为您生成的完整报告。")
        if len(history.messages) > 0 and history.messages[-1].type == 'ai':
            report_content = history.messages[-1].content
            text_only_report = re.sub("```mermaid\n(.*?)\n```", "\n[此处原为可视化图表]\n", report_content,
                                      flags=re.DOTALL)
            st.download_button(label="📥 下载完整报告 (.md)", data=text_only_report.encode('utf-8'),
                               file_name="我的职业路径规划报告.md", mime="text/markdown")


# ----------------------------------------------------------------
# --- 模式六：专业培养方案解析 (整合OCR的最终版) ---
# ----------------------------------------------------------------
def render_curriculum_mode(llm):
    st.header("模式六: 专业培养方案解析")
    st.markdown("---")
    history = get_session_history("curriculum_session")
    stage = st.session_state.get('curriculum_stage', 1)

    for msg in history.messages:
        avatar = "🧑‍💻" if isinstance(msg, HumanMessage) else "🤖"
        with st.chat_message(msg.type, avatar=avatar):
            if msg.type == 'ai' and "```mermaid" in msg.content:
                parts = msg.content.split("```mermaid")
                st.markdown(parts[0], unsafe_allow_html=True)
                mermaid_section = "```mermaid" + parts[1]
                mermaid_match = re.search("```mermaid\n(.*?)\n```", mermaid_section, re.DOTALL)
                if mermaid_match:
                    mermaid_code = mermaid_match.group(1)
                    st.subheader("重点课程学习路径图")
                    with st.container(border=True):
                        st_mermaid(mermaid_code.strip(), height="500px")
                after_diagram_content = mermaid_section.split("```")[-1]
                if after_diagram_content.strip():
                    st.markdown(after_diagram_content, unsafe_allow_html=True)
            else:
                st.markdown(msg.content, unsafe_allow_html=True)

    if stage == 1:
        st.info("请上传您专业的本科人才培养方案（PDF或TXT格式），AI学业导师将为您深度解析。")
        uploaded_file = st.file_uploader("点击此处上传文件...", type=['pdf', 'txt'], label_visibility="collapsed")

        if uploaded_file is not None:
            if st.button("第一步：分析人才培养方向", use_container_width=True, type="primary"):
                content = ""
                with st.spinner(f"正在读取文件 '{uploaded_file.name}'..."):
                    try:
                        import PyPDF2
                        uploaded_file.seek(0)
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        for page in pdf_reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                content += page_text + "\n\n"

                        if not content.strip():
                            st.info("快速读取失败，已自动切换至AI文字识别(OCR)模式，处理扫描件速度较慢，请稍候...")
                            import pytesseract
                            from pdf2image import convert_from_bytes

                            uploaded_file.seek(0)
                            images = convert_from_bytes(uploaded_file.read())
                            ocr_texts = []
                            for i, image in enumerate(images):
                                with st.spinner(f"正在识别第 {i + 1}/{len(images)} 页..."):
                                    text = pytesseract.image_to_string(image, lang='chi_sim+eng')
                                    ocr_texts.append(text)
                            content = "\n\n--- Page Break ---\n\n".join(ocr_texts)

                        if not content.strip():
                            st.error("无法从文件中提取有效文本内容，即使尝试了OCR也失败了。请检查文件是否损坏或过于模糊。")
                            st.stop()

                        st.session_state.curriculum_content = content
                        history.add_user_message("这是我的专业培养方案，请帮我分析。")

                        prompt = ChatPromptTemplate.from_template(
                            """
                            核心角色: 你是一位资深的大学学业导师和职业规划专家。
                            任务: 请严格按照以下结构，生成一份关于这份本科人才培养方案的分析报告。
                            **第一部分：人才培养方向分析报告**
                            1. **培养目标概括**: 精炼地总结该专业的核心培养目标。
                            2. **核心能力要求**: 根据“毕业要求”，提炼出学生需要掌握的3-4项最核心的能力。
                            **第二部分：建议的职业发展方向**
                            - 基于上述分析，特别是培养目标中提到的就业领域，提出 3-5 个具体的职业发展方向建议。
                            - 以项目符号列表的形式清晰呈现。
                            最后，请明确引导用户：“请从以上方向中选择一个您最感兴趣的，我将为您生成专属的学习路径规划图。”
                            培养方案全文如下: {curriculum_content}
                            """
                        )
                        chain = prompt | llm
                        with st.chat_message("ai", avatar="🤖"):
                            with st.spinner("AI导师正在深度分析培养方案..."):
                                response = st.write_stream(chain.stream({"curriculum_content": content}))
                                history.add_ai_message(response)
                        st.session_state.curriculum_stage = 2
                        st.rerun()

                    except Exception as e:
                        if "pytesseract" in str(e) or "pdf2image" in str(e):
                            st.error("错误：缺少OCR相关库。请运行 `pip install pytesseract pdf2image`。")
                        elif "Tesseract is not installed" in str(e) or "poppler" in str(e).lower():
                            st.error("错误：Tesseract OCR引擎或Poppler工具未安装或未在系统路径中。请参考说明完成安装。")
                        else:
                            st.error(f"处理文件时发生未知错误: {e}")
        else:
            st.button("第一步：分析人才培养方向", use_container_width=True, disabled=True)

    elif stage == 2:
        if user_input := st.chat_input("请输入您选择的职业方向..."):
            st.session_state.chosen_career = user_input
            history.add_user_message(user_input)
            prompt = ChatPromptTemplate.from_template(
                """
                核心角色: 你是一位资深的大学学业导师。
                任务: 用户选择了 **“{career_path}”** 作为职业方向。请为他生成一份重点专业科目学习规划。
                你的回答必须包含以下部分:
                1.  **学习路径规划说明**: 首先，简要阐述针对“{career_path}”方向，学习的重点和建议的先后顺序。
                2.  **学习路径关联图 (Mermaid)**:
                    -   创建一个 `graph TD` 类型的Mermaid流程图。
                    -   **【语法铁律】**: 如果课程名称（节点文本）中包含括号 `()` 或其他特殊符号，则**必须**将整个文本用双引号 `""` 括起来。例如：`C1["人际交往心理学(研讨课)"]`。
                    -   **必须进行颜色标注**: 将 **核心专业课** 节点背景色设为 `#D1E8FF` (淡蓝色)，将 **相关基础课** 节点背景色设为 `#FFF2CC` (淡黄色)。
                    -   在Mermaid代码块的 **最下方**，使用 `style` 命令来定义颜色。
                    -   在图表下方，必须添加图例说明。
                3.  **核心课程列表 (重要)**: 在图表和图例之后，请另起一行，并使用以下**一字不差的固定格式**列出所有被你识别为“核心专业课”（即淡蓝色节点）的课程名称。这是程序能否继续运行的关键。
                    格式:
                    ### 核心课程列表
                    - 课程A
                    - 课程B
                    - 课程C
                培养方案全文参考: {curriculum_content}
                """
            )
            chain = prompt | llm
            with st.chat_message("ai", avatar="🤖"):
                with st.spinner(f"正在为“{user_input}”方向规划学习路径..."):
                    response = st.write_stream(chain.stream({
                        "career_path": user_input,
                        "curriculum_content": st.session_state.curriculum_content
                    }))
                    history.add_ai_message(response)

                    # --- 优化的核心课程提取逻辑 ---
                    key_courses = []
                    if "### 核心课程列表" in response:
                        content_after_heading = response.split("### 核心课程列表")[1]
                        matches = re.findall(r"^\s*[-*]\s+(.*)", content_after_heading, re.MULTILINE)
                        if matches:
                            key_courses = [course.strip() for course in matches]

                    if key_courses:
                        st.session_state.key_courses_identified = key_courses
                    else:
                        st.session_state.key_courses_identified = None  # 确保失败时状态为空

            st.session_state.curriculum_stage = 3
            st.rerun()

    elif stage == 3:
        st.info("学习路径图已生成。现在，AI将为您详细解读其中的核心课程。")
        if st.button("第二步：生成核心课程教学目的报告", use_container_width=True, type="primary"):
            key_courses = st.session_state.get('key_courses_identified')
            if not key_courses:
                st.error("未能从上一步中识别出核心课程列表，请返回上一步重试。")
                st.stop()
            history.add_user_message(f"请为我详细解读这些核心课程：{', '.join(key_courses)}")
            prompt = ChatPromptTemplate.from_template(
                """
                核心角色: 你是一位专业的课程教学设计师。
                任务: 请为以下 **核心专业课程** 生成一份详细的教学目的与要求报告。
                核心课程列表: **{key_courses_list}**
                请严格按照以下格式，为列表中的 **每一门** 课程进行阐述:
                ### 课程名称：[例如：咨询心理学]
                -   **📖 知识目标**: 学生通过本课程将掌握哪些核心理论、概念和知识体系。
                -   **🛠️ 能力目标**: 本课程旨在培养学生的哪些具体技能。
                -   **🌟 素养目标**: 本课程如何帮助学生建立正确的价值观、职业道德或科学精神。
                你需要结合整个培养方案的上下文来进行推断和阐述。
                培养方案全文参考: {curriculum_content}
                """
            )
            chain = prompt | llm
            with st.chat_message("ai", avatar="🤖"):
                with st.spinner("正在生成核心课程的详细教学目的报告..."):
                    response = st.write_stream(chain.stream({
                        "key_courses_list": ", ".join(key_courses),
                        "curriculum_content": st.session_state.curriculum_content
                    }))
                    history.add_ai_message(response)
            st.session_state.curriculum_stage = 4
            st.rerun()

    elif stage == 4:
        st.success("🎉 专业培养方案解析已全部完成！希望这份详细的学业规划报告能为你的学习之旅点亮一盏明灯。")


def main():
    llm = get_llm_instance()
    if not llm:
        st.error("无法初始化语言模型，应用程序无法启动。请检查您的 API Key 设置。")
        st.stop()
    with st.sidebar:
        if st.session_state.get("current_mode", "menu") != "menu":
            if st.button("↩️ 返回主菜单"):
                st.session_state.clear()
                st.session_state.current_mode = "menu"
                st.rerun()
        st.markdown("---")
        st.caption("© 2025 智慧职业辅导 V14.3 (稳定版)")
    modes = {
        "menu": render_menu,
        "exploration": render_exploration_mode,
        "decision": render_decision_mode,
        "communication": render_communication_mode,
        "company_info": render_company_info_mode,
        "panoramic": render_panoramic_mode,
        "curriculum_analysis": render_curriculum_mode,
    }
    mode_func = modes.get(st.session_state.get("current_mode", "menu"), render_menu)
    if st.session_state.get("current_mode", "menu") == 'menu':
        mode_func()
    else:
        mode_func(llm)


if __name__ == "__main__":
    main()