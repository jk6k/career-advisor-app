import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time
from langchain_core.messages import HumanMessage, AIMessage

# --- Page Configuration (Must be the first Streamlit command) ---
# --- 页面配置 (必须是第一个 Streamlit 命令) ---
st.set_page_config(
    page_title="智慧化职业发展辅导系统",
    page_icon="💡",
    layout="wide"
)

# --- Custom CSS for Beautification ---
# --- 用于美化的自定义 CSS ---
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
    .stTextInput, .stTextArea {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization ---
# --- 初始化 ---
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# --- GLOBAL SYSTEM PERSONA from Design Document (Updated for Chinese Output)---
# --- 根据设计文档定义的全局系统角色 (已更新以限定中文输出) ---
GLOBAL_PERSONA = """
核心角色:你是一位智慧、专业且富有同理心的职业发展教练。
对话风格:你的语言应始终保持积极、鼓励和启发性。避免使用过于生硬或机械的语言,多使用引导性的提问来激发用户的思考。
核心目标:你的最终目标不是为用户提供唯一的“正确答案”,而是通过结构化的流程和富有洞察力的建议,赋予用户自主进行职业决策的能力。你要成为一个赋能者,而非一个决策者。
语言要求:你的所有回答都必须使用简体中文。
"""

# --- PROMPT DEFINITIONS based on Design Document (Completed) ---
# --- 基于设计文档的提示词定义 (已补全) ---
EXPLORATION_PROMPTS = {
    1: {
        "title": "阶段一：我是谁？",
        "prompt": "你好！我是一款职业目标规划辅助AI。我将通过一个经过验证的分析框架,引导你更具体、更系统地思考“职业目标是怎么来的”,并最终找到属于你自己的方向。\n\n让我们从核心开始,也就是“我”。请你用几个关键词或短句具体描述一下:\n\n1. 你的专业/个人兴趣点是什么?\n2. 你认为自己最擅长的三项能力是什么?\n3. 在未来的工作中,你最看重的是什么?"
    },
    2: {
        "title": "阶段二：我拥有什么平台和机会？",
        "prompt": "现在，我们来分析“我”所拥有的外部“平台与机会”。这能帮助你更客观地评估现状。\n\n请思考并回答：\n1. 从毕业院校/过往经历来看，你认为自己最大的优势平台是什么？\n2. 在你感兴趣的领域，你接触到的最前沿的机会或趋势是什么？\n3. 你的家庭或重要人际关系，能为你提供哪些支持？（情感、信息、资源等）"
    },
    3: {
        "title": "阶段三：我被什么所影响？",
        "prompt": "接下来，我们探讨一些需要持续“觉察”的因素。它们像“背景音”，深刻但不易察觉地影响着你的决策。\n\n请尝试描述：\n1. 你对“理想工作”的画像，主要受到了哪些人/信息源的影响？\n2. 当你畅想未来时，内心最深处的恐惧或担忧是什么？\n3. 在做选择时，你更倾向于规避风险，还是追求可能性？"
    },
    4: {
        "title": "阶段四：核心三角关系整合与决策模拟",
        "prompt": "非常棒的深入思考！现在，我们将“我是谁”、“我有什么”、“我受何影响”这三个核心进行整合。\n\n请尝试完成一个决策模拟：\n1. 基于前三部分的思考，请你构思出1-2个你认为“似乎可行”的职业发展方向。\n2. 想象你选择了其中一个方向，你预见到最大的挑战或困难是什么？\n3. 为了应对这个挑战，你现在最需要学习或提升的核心能力是什么？"
    },
    5: {
        "title": "阶段五：总结与行动",
        "prompt": "我们的探讨即将结束。最后一步，是“如何做到坚定而灵活”。\n\n请回答最后一个问题，将思考转化为行动：\n\n1. 为了验证或推进你在第四阶段构思的方向，你下周可以完成的第一个最小可行性动作是什么？（例如：和一位前辈交流、看一本书、学习一门课程的第一节等）"
    },
}


# --- LLM Initialization ---
# --- LLM 初始化 ---
@st.cache_resource
def get_llm_instance():
    """Initializes and returns the LLM instance, handling both local and deployed environments."""
    api_key = None
    # First, try to get the secret from Streamlit's secrets management (for deployment)
    try:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
    except (KeyError, FileNotFoundError):
        # If it fails (e.g., locally, no secrets.toml), fall back to environment variables
        api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        st.error("错误：未找到 DEEPSEEK_API_KEY。请在 Streamlit 云端后台或本地 .env 文件中设置它。")
        st.info(
            "如果您在本地运行，请确保在项目根目录创建一个名为 .env 的文件，并添加以下内容：\nDEEPSEEK_API_KEY='your_actual_api_key'")
        return None

    try:
        # --- CORRECTED MODEL AND BASE_URL ---
        # --- 修正了模型名称和接口地址，恢复到您原始的配置 ---
        llm = ChatOpenAI(
            model="deepseek-r1-250528",
            temperature=0.7,
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
        # Test call to ensure connectivity
        llm.invoke("Hello")
        return llm
    except Exception as e:
        st.error(f"初始化模型时出错: {e}")
        return None


# --- Session State Management ---
# --- 会话状态管理 ---
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
    """Retrieves or creates a chat history for a given session ID."""
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = ChatMessageHistory()
    return st.session_state.chat_history[session_id]


# --- UI Rendering Functions for Each Mode ---
# --- 各模式的 UI 渲染函数 ---

def render_menu():
    """Renders the main menu UI."""
    st.title("💡 智慧化职业发展辅导系统")
    st.markdown("---")
    st.subheader("请选择需要使用的功能模式：")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧭 职业目标探索", use_container_width=True):
            st.session_state.current_mode = "exploration"
            st.session_state.exploration_stage = 1
            st.session_state.chat_history['exploration_session'] = ChatMessageHistory()
            st.session_state.report_generated = False
            st.rerun()
        if st.button("🤔 家庭沟通模拟", use_container_width=True):
            st.session_state.current_mode = "communication"
            st.session_state.sim_started = False
            st.session_state.chat_history['communication_session'] = ChatMessageHistory()
            st.rerun()
    with col2:
        if st.button("⚖️ Offer决策分析", use_container_width=True):
            st.session_state.current_mode = "decision"
            st.rerun()
        if st.button("🏢 企业信息速览", use_container_width=True):
            st.session_state.current_mode = "company_info"
            st.rerun()


def render_exploration_mode(llm):
    """Renders the Career Goal Exploration mode UI and logic."""
    st.header("🧭 模式一: 职业目标探索")

    stage = st.session_state.exploration_stage

    # Display chat history first
    history = get_session_history("exploration_session")
    for msg in history.messages:
        avatar = "🧑‍💻" if msg.type == "human" else "🤖"
        with st.chat_message(msg.type, avatar=avatar):
            st.markdown(msg.content)

    # If exploration is finished, show report generation option
    if stage > 5:
        st.success("您已完成所有阶段的探索！现在，我可以为您生成一份综合报告。")
        if not st.session_state.report_generated:
            if st.button("✨ 生成我的职业探索报告"):
                with st.spinner("AI正在全面分析您的回答，生成专属报告..."):
                    # Format the entire chat history for the report prompt
                    full_conversation = "\n".join(
                        [f"{'用户' if isinstance(msg, HumanMessage) else 'AI教练'}: {msg.content}" for msg in
                         history.messages])

                    report_prompt_template = ChatPromptTemplate.from_template(GLOBAL_PERSONA + """
                    作为一名资深的职业发展教练，请根据以下用户与AI教练的完整对话记录，为用户撰写一份全面、深刻且富有启发性的职业探索总结报告。

                    报告需要遵循以下结构，并使用清晰的Markdown格式：

                    ### 1. 核心自我认知（我是谁？）
                    - 总结用户对自己专业兴趣、核心能力和职业价值观的认知。
                    - 提炼出用户最关键的个人特质和内在驱动力。

                    ### 2. 外部资源评估（我有什么？）
                    - 总结用户所拥有的平台优势、外部机会和人际支持网络。
                    - 分析这些资源如何为用户的职业发展提供可能性。

                    ### 3. 内在影响因素洞察（我受何影响？）
                    - 总结影响用户决策的深层因素，包括他人的影响、内心的担忧以及风险偏好。
                    - 点出用户在做选择时可能存在的思维惯性或盲点。

                    ### 4. 整合方向与潜在挑战（我的方向？）
                    - 总结用户初步构想的1-2个职业方向。
                    - 基于前面的分析，评估这些方向的合理性，并指出用户预见到的主要挑战。

                    ### 5. 下一步行动计划（我做什么？）
                    - 明确指出用户为自己设定的、可立即执行的最小行动步骤。
                    - 对这个行动计划的可行性给予鼓励和肯定。

                    ### 6. 综合建议
                    - 基于整体对话，提供1-2条核心建议，鼓励用户继续探索，并提醒他们关注的关键点。
                    - 结尾应积极、鼓舞人心，强调职业探索是一个持续的过程。

                    ---
                    以下是完整的对话记录:
                    {conversation_history}
                    ---
                    """)
                    report_chain = report_prompt_template | llm
                    report_response = report_chain.invoke({"conversation_history": full_conversation})
                    st.session_state.generated_report = report_response.content
                    st.session_state.report_generated = True
                    st.rerun()

        if st.session_state.report_generated:
            st.markdown("---")
            st.subheader("📄 您的个人职业探索报告")
            st.markdown(st.session_state.generated_report)
            st.info("希望这份报告能为您带来新的启发。您可以复制、保存这份报告，作为未来决策的参考。")

    # If exploration is ongoing
    else:
        st.info("此模式将通过五个阶段，引导您深入探索职业目标。")
        current_prompt_info = EXPLORATION_PROMPTS.get(stage)
        st.subheader(current_prompt_info["title"])

        # Display the current stage prompt if it's AI's turn (no human message yet for this stage)
        if len(history.messages) % 2 == 0:
            with st.chat_message("ai", avatar="🤖"):
                st.markdown(current_prompt_info["prompt"])

        meta_prompt = ChatPromptTemplate.from_messages([
            ("system", GLOBAL_PERSONA + """
            You are a thoughtful and insightful career planning coach. Your goal is to help the user think more deeply about their answers based on a five-stage framework.
            After the user answers the questions for a stage, your task is to:
            1. Acknowledge their response.
            2. Provide a brief, insightful comment or a thought-provoking follow-up question that connects their answer to the underlying principles of the framework. You should act as a suggestion provider, not just a data collector.
            3. Keep your feedback concise (2-3 sentences) and in Simplified Chinese.
            4. After providing your feedback, the program will automatically move to the next stage. So, you don't need to say "let's move on".
            Your response should add value and encourage deeper reflection.
            You are currently in Stage {current_stage} of the process.
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
            with st.spinner("AI正在分析您的回答并提供建议..."):
                chain_with_history.invoke(
                    {"input": user_input, "current_stage": stage},
                    config={"configurable": {"session_id": "exploration_session"}}
                )
                st.session_state.exploration_stage += 1
                st.rerun()


def render_decision_mode(llm):
    """Renders the Offer Decision Analysis mode UI and logic."""
    st.header("⚖️ 模式二: Offer决策分析")
    st.info("请输入两个Offer的关键信息，AI将为您生成一份结构化的对比分析报告。")

    meta_prompt = ChatPromptTemplate.from_template(GLOBAL_PERSONA + """
You are an expert career advisor. Your task is to conduct a structured analysis of two job offers for a user.
Offer A Details: {offer_a_details}
Offer B Details: {offer_b_details}

Please perform the following steps:
1. Create a Comparison Table: Generate a clear markdown table comparing the two offers side-by-side. Key comparison dimensions should include (but are not limited to): Company, Position, Salary/Compensation, Location, Career Growth Potential, and Work-Life Balance.
2. Pros and Cons Analysis: For each offer, list its main advantages (Pros) and disadvantages (Cons) based on the user's input and general career knowledge.
3. Recommendation and Key Questions: Provide a concluding recommendation. Do not make a definitive choice for the user, but suggest which offer might be more suitable based on different priorities (e.g., "If you prioritize immediate financial return, Offer A seems better..."). Finally, pose 1-2 key questions to help the user make their final decision.

Structure your entire response in clear, easy-to-read markdown.
""")
    chain = meta_prompt | llm

    col1, col2 = st.columns(2)
    with col1:
        offer_a = st.text_area("Offer A 关键信息", height=200, placeholder="例如: 公司名、职位、薪资、地点、优点、顾虑等")
    with col2:
        offer_b = st.text_area("Offer B 关键信息", height=200, placeholder="同样，包括公司名、职位、薪资、地点、优点、顾虑等")

    if st.button("✨ 生成对比分析报告", use_container_width=True):
        if not offer_a or not offer_b:
            st.warning("请输入两个Offer的信息后再生成报告。")
        else:
            with st.spinner("正在为您生成Offer分析报告..."):
                try:
                    response = chain.invoke({"offer_a_details": offer_a, "offer_b_details": offer_b})
                    st.markdown("---")
                    st.subheader("📋 Offer对比分析报告")
                    st.markdown(response.content)
                except Exception as e:
                    st.error(f"生成报告时出错: {e}")


def render_communication_mode(llm):
    """Renders the Family Communication Simulation mode UI and logic."""
    st.header("🤔 模式三: 家庭沟通模拟")

    # Setup section
    if not st.session_state.sim_started:
        st.info("在这里，AI可以扮演您的家人，帮助您练习如何沟通职业规划。")
        my_choice = st.text_input("首先, 请告诉我你想要和家人沟通的职业选择是什么?")
        family_concern = st.text_area("你认为他们主要的担忧会是什么?",
                                      placeholder="例如: 工作不稳定、不是铁饭碗、离家太远等")

        if st.button("🎬 开始模拟"):
            if not my_choice or not family_concern:
                st.warning("请输入您的职业选择和预想的家人担忧。")
            else:
                st.session_state.my_choice = my_choice
                st.session_state.family_concern = family_concern
                st.session_state.sim_started = True

                # Create the initial prompt for the AI to start the conversation
                initial_ai_prompt = f"孩子，关于你想做 '{my_choice}' 这个事，我有些担心。我主要是觉得它 '{family_concern}'。我们能聊聊吗？"
                get_session_history("communication_session").add_ai_message(initial_ai_prompt)
                st.rerun()

    # Chat section
    if st.session_state.sim_started:
        st.success(f"模拟开始！AI正在扮演担忧您选择 “{st.session_state.my_choice}” 的家人。")

        meta_prompt = ChatPromptTemplate.from_messages([
            ("system", GLOBAL_PERSONA + """
You are an AI role-playing as a user's parent. The user wants to practice a difficult conversation about their career choice.

Your Persona: You are a loving but concerned parent. Your primary concerns stem from what the user has described: '{family_concern}'. You want the best for your child, which to you means stability, security, and a respectable career path. You are skeptical of new or unconventional choices like '{my_choice}'.

Your Task:
1. Start the conversation from the parent's perspective, expressing your concern based on what you know.
2. Listen to the user's responses and react naturally. If they make a good point, you can be partially convinced but still raise other questions. If they are purely emotional, express your worry more strongly.
3. Your goal is NOT to be convinced easily. The goal is to provide a realistic simulation to help the user practice.
4. Keep your responses concise and in character.
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

        # Display chat history
        for msg in get_session_history("communication_session").messages:
            avatar = "🧑‍💻" if msg.type == "human" else "🧓"
            with st.chat_message(msg.type, avatar=avatar):
                st.markdown(msg.content)

        # Get user input
        if user_input := st.chat_input("你的回应:"):
            with st.spinner("..."):
                chain_with_history.invoke(
                    {"input": user_input, "my_choice": st.session_state.my_choice,
                     "family_concern": st.session_state.family_concern},
                    config={"configurable": {"session_id": "communication_session"}}
                )
                st.rerun()


def render_company_info_mode(llm):
    """Renders the Company Info Quick Look mode UI and logic."""
    st.header("🏢 模式四: 企业信息速览")
    st.info("请输入公司全名，AI将模拟网络抓取并为您生成一份核心信息速览报告。")

    meta_prompt = ChatPromptTemplate.from_template(GLOBAL_PERSONA + """
You are a professional business analyst AI. Your task is to generate a concise, structured summary of a company based on its name.
Company Name: {company_name}

Simulate that you have scraped the company's official website, recent news, and recruitment portals. Generate a report in clear markdown format that includes the following sections:
1.  **公司简介(Company Profile):** A brief overview of the company, its mission, and its industry positioning.
2.  **核心产品/业务(Core Products/Business):** A list or description of its main products, services, or business units.
3.  **近期动态(Recent Developments):** Summarize 2-3 recent significant news items, product launches, or strategic shifts.
4.  **热招岗位方向(Hot Recruitment Areas):** Based on simulated recruitment data, list 3-5 key types of positions the company is likely hiring for (e.g., "后端开发工程师", "产品经理-AI方向", "市场营销专员").

The information should be plausible and well-structured.
""")
    chain = meta_prompt | llm

    company_name = st.text_input("请输入公司名称:", placeholder="例如：阿里巴巴、腾讯、字节跳动")

    if st.button("🔍 生成速览报告", use_container_width=True):
        if not company_name:
            st.warning("请输入公司名称。")
        else:
            with st.spinner(f"正在生成关于 “{company_name}” 的信息报告..."):
                try:
                    response = chain.invoke({"company_name": company_name})
                    st.markdown("---")
                    st.subheader(f"📄 {company_name} - 核心信息速览")
                    st.markdown(response.content)
                except Exception as e:
                    st.error(f"生成报告时出错: {e}")


# --- Main App Logic ---
# --- 主应用逻辑 ---
def main():
    """Main function to run the Streamlit app."""
    llm = get_llm_instance()
    if not llm:
        st.stop()

    with st.sidebar:
        st.title("导航")
        if st.session_state.current_mode != "menu":
            if st.button("返回主菜单"):
                # Reset states before going to menu
                st.session_state.current_mode = "menu"
                st.session_state.exploration_stage = 1
                st.session_state.chat_history = {}
                if 'sim_started' in st.session_state:
                    del st.session_state.sim_started
                if 'my_choice' in st.session_state:
                    del st.session_state.my_choice
                if 'family_concern' in st.session_state:
                    del st.session_state.family_concern
                if 'report_generated' in st.session_state:
                    del st.session_state.report_generated
                if 'generated_report' in st.session_state:
                    del st.session_state.generated_report
                st.rerun()

    modes = {
        "menu": render_menu,
        "exploration": lambda: render_exploration_mode(llm),
        "decision": lambda: render_decision_mode(llm),
        "communication": lambda: render_communication_mode(llm),
        "company_info": lambda: render_company_info_mode(llm),
    }
    # Execute the function for the current mode
    modes[st.session_state.current_mode]()


if __name__ == "__main__":
    main()
