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
    .stApp { background-color: #F8F9FA; }
    h1, h2, h3, h4, h5, h6 { color: #212529; font-weight: 700; }
    h1 { font-size: 32px; }
    h2 { font-size: 28px; border-bottom: 2px solid #E9ECEF; padding-bottom: 0.4em; }
    h3 { font-size: 22px; }
    .st-emotion-cache-z5fcl4 { background-color: #FFFFFF; border-radius: 12px; padding: 28px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #E9ECEF; transition: transform 0.3s ease, box-shadow 0.3s ease; }
    .st-emotion-cache-z5fcl4:hover { transform: translateY(-5px); box-shadow: 0 12px 24px rgba(0,0,0,0.08); }
    .stButton>button { border-radius: 8px; border: none; color: white; font-weight: 500; padding: 12px 24px; background-image: linear-gradient(135deg, #5D9CEC 0%, #4A90E2 100%); transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(74, 144, 226, 0.2); }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(74, 144, 226, 0.3); }
    .stChatMessage { border-radius: 12px; border: 1px solid #E9ECEF; background-color: #FFFFFF; padding: 16px; margin-bottom: 1rem; }
    .stChatInputContainer { position: sticky; bottom: 0; background-color: #FFFFFF; padding: 12px 0px; border-top: 1px solid #E9ECEF; box-shadow: 0 -4px 12px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# --- 初始化 ---
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# --- 全局系统角色 (简体中文) ---
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
        st.error(f"错误：未找到 {key_name}。请在 Streamlit Cloud Secrets 或本地 .env 文件中设置它。")
        return None
    try:
        llm = ChatOpenAI(model="deepseek-r1-250528", temperature=0.7, api_key=api_key,
                         base_url="https://ark.cn-beijing.volces.com/api/v3")
        llm.invoke("Hello");
        return llm
    except Exception as e:
        st.error(f"初始化模型时出错: {e}"); return None


# --- 会话状态管理 ---
def init_session_state():
    defaults = {"current_mode": "menu", "chat_history": {}, "exploration_stage": 1, "sim_started": False,
                "debrief_requested": False, "panoramic_stage": 1, "user_profile": None, "chosen_professions": None,
                "chosen_region": None}
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value


init_session_state()


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in st.session_state.chat_history: st.session_state.chat_history[session_id] = ChatMessageHistory()
    return st.session_state.chat_history[session_id]


# --- UI 渲染函数 ---
def render_menu():
    st.title("✨ 智慧化职业发展辅导系统")
    st.markdown("---");
    st.subheader("欢迎使用！请选择一项功能开始探索：");
    st.write("")
    modes_config = [
        ("exploration", ":compass: 职业目标探索", "通过“我-社会-家庭”框架，系统性地探索内在动机与外在机会。"),
        ("panoramic", ":globe_with_meridians: 职业路径全景规划",
         "从您的核心能力出发，连接职业、企业、地区与产业链，生成您的个人发展蓝图。"),
        ("decision", ":balance_scale: Offer 决策分析", "结构化对比多个Offer，获得清晰的决策建议。"),
        ("company_info", ":office: 企业信息速览", "快速了解目标公司的核心业务、近期动态与热招方向。"),
        ("communication", ":family: 家庭沟通模拟", "与AI扮演的家人进行对话，安全地练习如何表达您的职业选择。")
    ]
    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        with st.container(border=True):
            st.subheader(modes_config[0][1]);
            st.caption(modes_config[0][2])
            if st.button("开始探索", use_container_width=True, key=f"menu_{modes_config[0][0]}"):
                st.session_state.current_mode = modes_config[0][0];
                st.rerun()
        st.write("")
        with st.container(border=True):
            st.subheader(modes_config[1][1]);
            st.caption(modes_config[1][2])
            if st.button("开始规划", use_container_width=True, key=f"menu_{modes_config[1][0]}"):
                st.session_state.current_mode = modes_config[1][0];
                st.rerun()
    with col2:
        for mode_key, title, caption in modes_config[2:]:
            with st.container(border=True):
                st.subheader(title);
                st.caption(caption)
                if st.button(f"开始{title.split(' ')[1][:2]}", use_container_width=True, key=f"menu_{mode_key}"):
                    st.session_state.current_mode = mode_key;
                    st.rerun()
            st.write("")


def render_exploration_mode(llm):
    st.header("模式一: 职业目标探索")
    history = get_session_history("exploration_session")
    stage = st.session_state.get('exploration_stage', 1)
    for msg in history.messages:
        avatar = "🧑‍💻" if isinstance(msg, HumanMessage) else "🤖"
        st.chat_message(msg.type, avatar=avatar).markdown(msg.content, unsafe_allow_html=True)

    # 【V10.1 核心修正】恢复了原始版本中完整、清晰的引导文案和问题
    prompts = {
        1: {
            "title": "> **第一阶段：分析“我”(可控因素)**\n> \n> 你好！我将引导你使用“职业目标缘起分析框架”，从“我”、“社会”、“家庭”三个核心维度，系统性地探索你的职业方向。\n> \n> 首先，我们来分析“我”这个核心。请在下方回答：",
            "questions": ["1. 你的专业是什么？你对它的看法如何？", "2. 你的学校或过往经历，为你提供了怎样的平台与基础？"],
            "button_text": "提交关于“我”的分析"
        },
        2: {
            "title": "> **第二阶段：分析“社会”(外部机会)**\n> \n> 好的，我们盘点了“我”的基础。接着，我们来分析外部的“社会”因素。请思考：",
            "questions": ["1. 你观察到当下有哪些你感兴趣的社会或科技趋势？（例如：AI、大健康、可持续发展等）",
                          "2. 根据你的观察，这些趋势可能带来哪些新的行业或职位机会？",
                          "3. 在你过往的经历中，有没有一些偶然的机缘或打工经验，让你对某个领域产生了特别的了解？"],
            "button_text": "提交关于“社会”的分析"
        },
        3: {
            "title": "> **第三阶段：觉察“家庭”(环境影响)**\n> \n> 接下来，我们来探讨需要持续“觉察”的“家庭”与环境影响。请描述：",
            "questions": ["1. 你的家庭或重要亲友，对你的职业有什么样的期待？",
                          "2. 有没有哪位榜样对你的职业选择产生了影响？",
                          "3. 你身边的“圈子”（例如朋友、同学）主要从事哪些工作？这对你有什么潜在影响？"],
            "button_text": "提交关于“家庭”的分析"
        },
    }

    if stage in prompts:
        config = prompts[stage]
        st.markdown(config["title"])
        with st.form(f"stage{stage}_form"):
            responses = [st.text_area(q, height=100) for q in config["questions"]]
            if st.form_submit_button(config["button_text"], use_container_width=True):
                if all(responses):
                    input_text = f"### 关于阶段 {stage} 的回答\n\n" + "\n\n".join(
                        [f"**{q}**\n{r}" for q, r in zip(config["questions"], responses)])
                    history.add_user_message(input_text);
                    st.session_state.exploration_stage += 1;
                    st.rerun()
                else:
                    st.warning("请完整填写所有问题的回答。")
    elif stage == 4:
        st.markdown("> **第四阶段：AI 智慧整合与行动计划**")
        with st.chat_message("ai", avatar="🤖"):
            full_conversation = "\n\n".join([msg.content for msg in history.messages if isinstance(msg, HumanMessage)])
            stage4_prompt = ChatPromptTemplate.from_template(
                GLOBAL_PERSONA + "作为一名智慧且富有洞察力的职业发展教练，请严格根据以下用户在“我”、“社会”、“家庭”三个阶段的完整回答，为用户生成一份整合分析与建议报告。报告必须包含以下三个部分，并使用清晰的Markdown格式：\n\n### 1. 初步决策方向建议\n- ...\n\n### 2. 预期“收入”分析\n- ...\n\n### 3. 第一个“行动”建议\n- ...\n\n---\n以下是用户的完整回答: \n{conversation_history}\n---")
            stage4_chain = stage4_prompt | llm
            with st.spinner("AI教练正在全面分析您的回答..."):
                response_content = st.write_stream(stage4_chain.stream({"conversation_history": full_conversation}))
            history.add_ai_message(response_content)
        st.session_state.exploration_stage = 5;
        st.rerun()
    elif stage == 5:
        st.markdown(
            "> AI教练已根据您的回答，为您提供了一份整合分析与建议。这份报告是为您量身打造的起点，而非终点。\n>\n> 请仔细阅读报告，然后回答最后一个、也是最重要的问题：\n> **您自己决定要采取的、下周可以完成的第一个具体行动是什么？**")
        if user_input := st.chat_input("请在此输入您的最终行动计划..."):
            history.add_user_message(user_input);
            st.session_state.exploration_stage = 6;
            st.rerun()
    elif stage == 6:
        st.success("恭喜！您已完成本次探索的全过程。")


def render_decision_mode(llm):
    st.header("模式二: Offer 决策分析")
    with st.container(border=True):
        st.info("请输入两个Offer的关键信息，AI将为您生成一份结构化的对比分析报告。")
        chain = ChatPromptTemplate.from_template(
            GLOBAL_PERSONA + "You are an expert career advisor. Your task is to conduct a structured analysis of two job offers for a user. Offer A Details: {offer_a_details}. Offer B Details: {offer_b_details}. User Priorities: {user_priorities_sorted_list}. Please create a markdown report with: 1. Comparison Table. 2. Priority Matching Analysis. 3. Pros and Cons. 4. Risk Alert. 5. Recommendation.") | llm
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
                                         help="您选择的第一个选项代表您最看重的因素。")
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
            my_choice = st.text_input("首先, 请告诉我你想要和家人沟通的职业选择是什么?")
            family_concern = st.text_area("你认为他们主要的担忧会是什么?",
                                          placeholder="例如: 工作不稳定、不是铁饭碗、离家太远等")
            if st.button("开始模拟"):
                if not my_choice or not family_concern:
                    st.warning("请输入您的职业选择和预想的家人担忧。")
                else:
                    st.session_state.my_choice = my_choice;
                    st.session_state.family_concern = family_concern
                    st.session_state.sim_started = True;
                    st.session_state.debrief_requested = False
                    st.session_state.chat_history['communication_session'] = ChatMessageHistory()
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
                                                                      GLOBAL_PERSONA + f"""现在，你将扮演一个关心孩子但思想略显传统的家人。- 你的核心担忧是: "{st.session_state.family_concern}"- 你的对话目标是：反复确认孩子是否考虑清楚了这些担忧，而不是轻易被说服。"""),
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
                    GLOBAL_PERSONA + "你现在切换回职业发展教练的角色。对以下沟通记录进行复盘，必须包含：\n\n### 1. 沟通亮点\n\n### 2. 可优化点\n\n### 3. 具体话术建议\n\n---\n对话记录:\n{conversation_history}\n---")
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
        st.info("请输入公司全名，AI将为您生成一份核心信息速览报告。")
        chain = ChatPromptTemplate.from_template(
            GLOBAL_PERSONA + "You are a professional business analyst AI. Your task is to provide a concise and structured overview of a given company. Company Name: {company_name}. Please structure your report in markdown with these sections: ### 1. 公司简介, ### 2. 近期动态与新闻, ### 3. 企业文化与价值观, ### 4. 热门招聘方向.") | llm
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

    meta_prompt_template = GLOBAL_PERSONA + """
    You are an expert career strategist... (指令同V9.3版本)
    """
    chain = ChatPromptTemplate.from_template(meta_prompt_template) | llm
    if stage == 1:
        st.markdown("> 你好！我是你的职业路径规划助手。让我们从认识你自己开始。")
        with st.form("profile_form"):
            st.subheader("请根据以下五个维度，描述你的“核心能力”：")
            edu = st.text_area("学历背景", placeholder="你的专业、学位、以及相关的核心课程")
            skills = st.text_area("核心技能", placeholder="你最擅长的3-5项硬技能或软技能")
            exp = st.text_area("相关经验", placeholder="相关的实习、工作项目、或个人作品集")
            char = st.text_area("品行特质", placeholder="你认为自己最重要的职业品行或工作风格")
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


def main():
    llm = get_llm_instance()
    if not llm: st.error("无法初始化语言模型，应用程序无法启动。请检查您的 API Key 设置。"); st.stop()
    with st.sidebar:
        if st.session_state.current_mode != "menu":
            if st.button("↩️ 返回主菜单"):
                st.session_state.clear();
                st.session_state.current_mode = "menu";
                st.rerun()
        st.markdown("---");
        st.caption("© 2025 智慧职业辅导 V10.1")
    modes = {"menu": render_menu, "exploration": render_exploration_mode, "decision": render_decision_mode,
             "communication": render_communication_mode, "company_info": render_company_info_mode,
             "panoramic": render_panoramic_mode}
    mode_func = modes.get(st.session_state.current_mode, render_menu)
    if st.session_state.current_mode == 'menu':
        mode_func()
    else:
        mode_func(llm)


if __name__ == "__main__":
    main()