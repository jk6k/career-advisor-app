import os
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
import time
import textwrap
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Explicitly disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# --- PROMPT DEFINITIONS for Mode 1: Exploration ---
PROMPTS = {
    1: {
        "title": "阶段一：启动与导入 —— “我是谁？”",
        "prompt": """你好！我是一款职业目标规划辅助AI。我将通过一个经过验证的分析框架，引导你更具体、更系统地思考‘职业目标是怎么来的’，并最终找到属于你自己的方向。

让我们从核心开始，也就是‘我’。请你用几个关键词或短句具体描述一下：

1. 你的专业/个人兴趣点是什么？
2. 你认为自己最擅长的三项能力是什么？
3. 在未来的工作中，你最看重的是什么？(可多选)"""
    },
    2: {
        "title": "阶段二：可分析因素解构 —— “我拥有什么平台和机会？”",
        "prompt": """现在，我们来分析一些你可以阶段性审视的外部因素。

1. 关于【大学平台】:
   * 你所在的大学或专业，在哪些领域被认为是优势学科？
   * 通常有哪些类型的企业会来你们学校或学院进行招聘？这给了你什么具体的启发？
   * 你所在的城市为你提供了哪些独特的产业机会？

2. 关于【趋势】:
   * 你注意到了哪些正在发生的社会或技术趋势？
   * 请具体思考，这些趋势可能会催生出哪些与你的专业或兴趣相关的**新行业**或**新岗位**？

3. 关于【机缘】:
   * 回想一下，是否有某次偶然的经历让你对某个职业领域产生了新的兴趣？"""
    },
    3: {
        "title": "阶段三：需持续觉察因素的挖掘 —— “我被什么所影响？”",
        "prompt": """接下来，我们探讨一些需要持续‘觉察’的、更感性的影响因素。

1. 关于【家庭与社会期待】:
   * 你的长辈或家庭对你的职业有什么**具体**的期望吗？
   * 你身边的朋友或‘圈子’，他们的职业规划是怎样的？这对你产生了什么影响？

2. 关于【榜样】:
   * 是否有你非常敬佩的某个榜样人物？
   * 你欣赏的是他/她的**职业本身**，还是他/她工作带来的某种**状态**或其**‘周边’**价值？"""
    },
    4: {
        "title": "阶段四：核心三角关系整合与决策模拟",
        "prompt": """非常棒的深入思考！现在，让我们把这些碎片化的信息整合起来。

根据我们之前的对话，我为你做一个简单的回顾与总结：
{history}

**决策模拟：**
1. 结合以上所有信息，请尝试构思1-2个可能的职业方向。
2. 针对你构思的职业方向，我们来思考一个非常现实的问题：**‘如何(收入)’**。你期望的回报是什么？
3. 如果你的理想方向与家庭期待存在矛盾，你认为可以收集哪些信息来和家人开一次有理有据的‘家庭会议’呢？"""
    },
    5: {
        "title": "阶段五：总结与行动 —— “如何做到坚定而灵活？”",
        "prompt": """我们的探讨即将结束。职业规划不是一次性的终点，而是一个持续优化的过程。核心是达到**‘坚定而灵活’**的状态。

请记住，大学阶段是进行职业探索和‘试错’成本最低的时期。为了验证我们今天讨论出的职业方向，你可以立即采取的**第一个小步骤**是什么？

* A. 通过实习或勤工俭学去亲身体验？
* B. 找一位该领域的学长学姐或从业者深入交流？
* C. 学习一项与该方向相关的具体技能？
* D. 其他：__________

请选择并具体描述你的行动计划。"""
    }
}


# --- Helper Functions ---
def print_formatted(text, prefix="AI: "):
    """Helper function to print text with wrapping."""
    wrapped_text = textwrap.fill(text, width=80, subsequent_indent='    ', replace_whitespace=False)
    print(f"\n{prefix}{wrapped_text}\n")


def get_llm_instance():
    """Initializes and returns the LLM instance."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("\n" + "─" * 80)
        print(">>>>> 错误：未找到API密钥! <<<<<")
        print("\n请确认您的项目文件夹中包含 .env 文件，并且该文件中已设置 DEEPSEEK_API_KEY。")
        print("─" * 80 + "\n")
        return None

    try:
        llm = ChatOpenAI(
            model="deepseek-r1-250528",
            temperature=0.7,
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
        print("正在连接火山方舟（VolcEngine Ark）API...")
        llm.invoke("Hello")
        print("连接成功！")
        return llm
    except Exception as e:
        print("\n" + "─" * 80)
        print(">>>>> 初始化模型时出错! <<<<<")
        print(f"Error Details: {e}")
        return None


# --- Mode 1: Career Goal Exploration ---
def run_exploration_mode(llm):
    """Runs the 5-stage career exploration conversation."""
    store = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    current_stage = 1
    while current_stage <= len(PROMPTS):
        stage_info = PROMPTS[current_stage]
        title = stage_info["title"]
        system_prompt = stage_info["prompt"]

        if current_stage == 4:
            history_messages = get_session_history("exploration_session").messages
            history_summary = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in history_messages])
            final_system_prompt = system_prompt.format(history=history_summary)
        else:
            final_system_prompt = system_prompt

        meta_prompt = f"""You are a thoughtful and insightful career planning coach. Your goal is to help the user think more deeply about their answers based on a five-stage framework. After the user answers the questions for a stage, provide a brief, insightful comment or a thought-provoking follow-up question that connects their answer to the underlying principles of the framework. Keep your feedback concise (2-3 sentences).

You are currently in Stage {current_stage} of the process. The user is answering the following questions:
{final_system_prompt}
"""
        prompt = ChatPromptTemplate.from_messages(
            [("system", meta_prompt), MessagesPlaceholder(variable_name="history"), ("human", "{input}")])
        chain = RunnableWithMessageHistory(prompt | llm, get_session_history, input_messages_key="input",
                                           history_messages_key="history")

        print("─" * 80)
        print(f"                                   {title}")
        print("─" * 80)

        if current_stage == 1:
            print_formatted(system_prompt, prefix="")

        print("\n💡 请针对以上问题，将你的想法和思考一次性输入，然后按Enter键。")
        user_input = input("你的回答: ")

        if user_input.lower() in ['quit', 'exit']:
            print_formatted("好的，对话结束。正在返回主菜单...")
            break

        print("\n[AI正在分析您的回答并提供建议，请稍候...]")
        response = chain.invoke({"input": user_input}, config={"configurable": {"session_id": "exploration_session"}})
        print_formatted(response.content)

        time.sleep(2)
        current_stage += 1

        if current_stage <= len(PROMPTS):
            print_formatted(PROMPTS[current_stage]["prompt"], prefix="")

    print("─" * 80)
    print("职业目标探索流程已完成。正在返回主菜单...")
    print("─" * 80)


# --- Mode 2: Offer Decision Support ---
def run_decision_support_mode(llm):
    """Guides the user to analyze and compare job offers."""
    print("\n" + "─" * 80)
    print("                                 模式二：Offer决策分析")
    print("─" * 80)
    print_formatted("你好！当你手握多个Offer犹豫不决时，我可以通过结构化的方式，帮助你理清思路，做出更适合自己的选择。")

    offer_a_details = input("📄 请输入 Offer A 的关键信息 (例如：公司名、职位、薪资、地点、优点、顾虑等):\n")
    offer_b_details = input("\n📄 请输入 Offer B 的关键信息 (同样，包括公司名、职位、薪资、地点、优点、顾虑等):\n")

    print("\n[AI正在为您生成对比分析报告，请稍候...]")

    prompt_text = f"""You are an expert career advisor. Your task is to conduct a structured analysis of two job offers for a user.

    Offer A Details: {offer_a_details}
    Offer B Details: {offer_b_details}

    Please perform the following steps:
    1.  **Create a Comparison Table**: Generate a clear markdown table comparing the two offers side-by-side. Key comparison dimensions should include (but are not limited to): Company, Position, Salary/Compensation, Location, Career Growth Potential, and Work-Life Balance.
    2.  **Pros and Cons Analysis**: For each offer, list its main advantages (Pros) and disadvantages (Cons) based on the user's input and general career knowledge.
    3.  **Recommendation and Key Questions**: Provide a concluding recommendation. Do not make a definitive choice for the user, but suggest which offer might be more suitable based on different priorities (e.g., "If you prioritize immediate financial return, Offer A seems better, but if long-term growth is your goal, Offer B has more potential."). Finally, pose 1-2 key questions to help the user make their final decision.

    Structure your entire response in clear, easy-to-read markdown.
    """
    prompt = ChatPromptTemplate.from_messages([("human", prompt_text)])
    chain = prompt | llm
    response = chain.invoke({})
    print_formatted(response.content)
    input("\n分析已完成。按Enter键返回主菜单...")


# --- Mode 3: Family Communication Simulation ---
def run_communication_simulation_mode(llm):
    """Simulates a conversation with a family member about career choices."""
    print("\n" + "─" * 80)
    print("                              模式三：家庭沟通模拟")
    print("─" * 80)
    print_formatted(
        "你好！和家人沟通职业规划有时会遇到困难。在这里，我可以扮演你的家人（比如父亲或母亲），你可以安全地练习如何表达自己的想法，并应对可能出现的担忧和问题。")

    my_choice = input("🗣️ 首先，请告诉我你想要和家人沟通的职业选择是什么？\n")
    family_concern = input("\n🤔 你认为他们主要的担忧会是什么？(例如：工作不稳定、不是铁饭碗、离家太远等)\n")

    meta_prompt = f"""You are an AI role-playing as a user's parent. The user wants to practice a difficult conversation about their career choice.

    **Your Persona**: You are a loving but concerned parent. Your primary concerns stem from what the user has described: '{family_concern}'. You want the best for your child, which to you means stability, security, and a respectable career path. You are skeptical of new or unconventional choices.

    **Your Task**:
    1.  Start the conversation from the parent's perspective, expressing your concern based on what you know.
    2.  Listen to the user's responses and react naturally. If they make a good point, you can be partially convinced but still raise other questions. If they are purely emotional, express your worry more strongly.
    3.  Your goal is NOT to be convinced easily. The goal is to provide a realistic simulation to help the user practice.
    4.  Keep your responses concise and in character.

    Let's begin the simulation. You will speak first.
    """

    store = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
            # Start the conversation with the AI's (parent's) first line
            initial_ai_response = llm.invoke(meta_prompt)
            store[session_id].add_ai_message(initial_ai_response.content)
        return store[session_id]

    # Initialize the session and get the first message
    get_session_history("sim_session")
    initial_message = store["sim_session"].messages[0].content
    print_formatted(initial_message, prefix="AI (扮演家长):")

    prompt = ChatPromptTemplate.from_messages([
        ("system", meta_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    chain = RunnableWithMessageHistory(prompt | llm, get_session_history, input_messages_key="input",
                                       history_messages_key="history")

    while True:
        user_input = input("你的回应: ")
        if user_input.lower() in ['quit', 'exit']:
            print_formatted("好的，模拟结束。正在返回主菜单...")
            break

        print("\n[AI(家长)正在思考如何回应...]")
        response = chain.invoke({"input": user_input}, config={"configurable": {"session_id": "sim_session"}})
        print_formatted(response.content, prefix="AI (扮演家长):")


# --- Mode 4: Company Info Quick Look ---
def run_company_info_mode(llm):
    """Simulates scraping and summarizing company information."""
    print("\n" + "─" * 80)
    print("                               模式四：企业信息速览")
    print("─" * 80)
    print_formatted("你好！想快速了解一个公司吗？请输入公司全名，我将模拟抓取并为你生成一份核心信息速览报告。")

    company_name = input("🏢 请输入公司名称:\n")

    print(f"\n[正在模拟抓取 {company_name} 的相关信息并生成报告...]")

    prompt_text = f"""You are a professional business analyst AI. Your task is to generate a concise, structured summary of a company based on its name.

    Company Name: {company_name}

    Simulate that you have scraped the company's official website, recent news, and recruitment portals. Generate a report in clear markdown format that includes the following sections:

    1.  **公司简介 (Company Profile)**: A brief overview of the company, its mission, and its industry positioning.
    2.  **核心产品/业务 (Core Products/Business)**: A list or description of its main products, services, or business units.
    3.  **近期动态 (Recent Developments)**: Summarize 2-3 recent significant news items, product launches, or strategic shifts.
    4.  **热招岗位方向 (Hot Recruitment Areas)**: Based on simulated recruitment data, list 3-5 key types of positions the company is likely hiring for (e.g., "后端开发工程师", "产品经理-AI方向", "市场营销专员").

    The information should be plausible and well-structured.
    """

    prompt = ChatPromptTemplate.from_messages([("human", prompt_text)])
    chain = prompt | llm
    response = chain.invoke({})
    print_formatted(response.content)
    input("\n报告已生成。按Enter键返回主菜单...")


# --- Main Application ---
if __name__ == "__main__":
    print("AI职业规划助手原型启动 (使用火山方舟DeepSeek模型)...")

    llm_instance = get_llm_instance()

    if llm_instance:
        while True:
            print("\n" + "=" * 35 + " 主菜单 " + "=" * 35)
            print("请选择需要使用的功能模式：\n")
            print("  1. 职业目标探索 (通过深度对话，进行自我分析与规划)")
            print("  2. Offer决策分析 (输入多个Offer，获取结构化对比建议)")
            print("  3. 家庭沟通模拟 (扮演家人角色，练习职业选择的沟通)")
            print("  4. 企业信息速览 (输入公司名，快速了解核心情况)")
            print("\n  0. 退出程序")
            print("=" * 80)

            choice = input("\n请输入选项 (0-4): ")

            if choice == '1':
                run_exploration_mode(llm_instance)
            elif choice == '2':
                run_decision_support_mode(llm_instance)
            elif choice == '3':
                run_communication_simulation_mode(llm_instance)
            elif choice == '4':
                run_company_info_mode(llm_instance)
            elif choice == '0':
                print_formatted("感谢使用，再见！")
                break
            else:
                print("无效输入，请输入 0 到 4 之间的数字。")

            time.sleep(1)  # A brief pause before showing the menu again
