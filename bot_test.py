import os
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# --- UPDATED IMPORT for ChatMessageHistory ---
# The warning suggests importing from langchain_community.chat_message_histories
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
# --- UPDATED IMPORTS ---
# We will use the generic ChatOpenAI client which allows specifying a custom API endpoint.
from langchain_openai import ChatOpenAI
import time
import textwrap
# ADDED: Import the dotenv library to load the .env file
from dotenv import load_dotenv

# ADDED: Load environment variables from the .env file
load_dotenv()

# --- FIX for LangSmith Error ---
# Explicitly disable LangSmith tracing to prevent connection errors
# if the user doesn't have it configured.
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# --- 1. 从设计文档中提取各阶段的提示词 ---
# --- UPDATED: Rephrased all prompts to give the AI its own voice ---
PROMPTS = {
    1: {
        "title": "阶段一：启动与导入 —— “我是谁？”",
        "prompt": """你好！我是一款职业目标规划辅助AI。我将通过一个经过验证的分析框架，引导你更具体、更系统地思考‘职业目标是怎么来的’，并最终找到属于你自己的方向。

让我们从核心开始，也就是‘我’。请你用几个关键词或短句具体描述一下：

1. 你的专业/个人兴趣点是什么？ (例如：法律中的商法、AI绘画、市场营销案例分析)
2. 你认为自己最擅长的三项能力是什么？ (例如：逻辑分析、公开演讲、视频剪辑)
3. 在未来的工作中，你最看重的是什么？(可多选) (例如：高薪酬、工作稳定、能帮助他人、持续学习、创造性强)"""
    },
    2: {
        "title": "阶段二：可分析因素解构 —— “我拥有什么平台和机会？”",
        "prompt": """现在，我们来分析一些你可以阶段性审视的外部因素。

1. 关于【大学平台】:
   * 你所在的大学或专业，在哪些领域被认为是优势学科？(例如，一所以法学见长的大学，可能在仲裁、警务等领域有独特的就业优势)
   * 通常有哪些类型的企业会来你们学校或学院进行招聘？这给了你什么具体的启发？
   * 你所在的城市为你提供了哪些独特的产业机会？(例如，内蒙古鄂尔多斯的煤炭产业)

2. 关于【趋势】:
   * 你注意到了哪些正在发生的社会或技术趋势？(例如：AI发展、老龄化、新能源)
   * 请具体思考，这些趋势可能会催生出哪些与你的专业或兴趣相关的**新行业**或**新岗位**？(例如，老龄化趋势让养老产业的需求变得更旺盛)

3. 关于【机缘】:
   * 回想一下，是否有某次偶然的经历（如一次讲座、与某人的谈话、一个项目）让你对某个职业领域产生了新的兴趣？有时候，一次不经意的接触就可能开启一扇职业大门。"""
    },
    3: {
        "title": "阶段三：需持续觉察因素的挖掘 —— “我被什么所影响？”",
        "prompt": """接下来，我们探讨一些需要持续‘觉察’的、更感性的影响因素。这部分讨论可能比较个人化，请坦诚地面对自己的想法。

1. 关于【家庭与社会期待】:
   * 你的长辈或家庭对你的职业有什么**具体**的期望吗？(例如：希望你考公务员，进入某个特定行业)
   * 你身边的朋友或‘圈子’，他们的职业规划是怎样的？这对你产生了什么影响？

2. 关于【榜样】:
   * 是否有你非常敬佩的某个榜样人物（公众人物、老师、学长学姐等）？
   * 你欣赏的是他/她的**职业本身**，还是他/她工作带来的某种**状态**或其**‘周边’**价值（如社会影响力、生活方式）？"""
    },
    4: {
        "title": "阶段四：核心三角关系整合与决策模拟",
        "prompt": """非常棒的深入思考！现在，让我们把这些碎片化的信息整合起来，看看‘我’、‘社会’、‘家庭’这个核心三角关系。

根据我们之前的对话，我为你做一个简单的回顾与总结：
{history}

**决策模拟：**
1. 结合以上所有信息，请尝试构思1-2个可能的职业方向。
2. 针对你构思的职业方向，我们来思考一个非常现实的问题：**‘如何(收入)’**。你期望的回报是什么？是直接的金钱，还是包括社会价值、个人成就感在内的间接回报？
3. 如果你的理想方向与家庭期待存在矛盾，直接对抗往往效果不佳。一个更有效的方法是，通过系统的分析，拿出数据和事实进行沟通。你认为可以收集哪些信息来和家人开一次有理有据的‘家庭会议’呢？"""
    },
    5: {
        "title": "阶段五：总结与行动 —— “如何做到坚定而灵活？”",
        "prompt": """我们的探讨即将结束。职业规划不是一次性的终点，而是一个持续优化的过程。核心是达到**‘坚定而灵活’**的状态：基于理性分析而来的方向是‘坚定’的，但随时准备根据环境变化和新的认知进行‘灵活’调整。

请记住，大学阶段是进行职业探索和‘试错’成本最低的时期。为了验证我们今天讨论出的职业方向，你可以立即采取的**第一个小步骤**是什么？

* A. 通过实习或勤工俭学去亲身体验？
* B. 找一位该领域的学长学姐或从业者深入交流？
* C. 学习一项与该方向相关的具体技能（如视频剪辑、数据分析）？
* D. 其他：__________

请选择并具体描述你的行动计划。记住，‘机缘来自于接触’，行动起来才能让规划变得真正具体！"""
    }
}

# --- ADDED: Test Case Data ---
TEST_ANSWERS = [
    "我的专业是法学，但我对AI技术和它如何影响社会很感兴趣。我擅长资料研究、逻辑分析和写作。我希望未来工作能有挑战性，并且能持续学习新知识。",
    "我们学校的法学院很有名，经常有律所和法院来招聘。但我所在的城市也是一个科技中心，有很多AI创业公司。我注意到AI正在颠覆法律行业，比如AI合同审查工具，这可能是一个新的方向。我最近参加了一个关于'法律科技'的讲座，感觉很有启发。",
    "我的家人希望我能成为一名稳定的律师或考公务员。我敬佩的一位榜样是一位用技术创业的律师，他创办了一个在线法律服务平台，我觉得他不仅懂法律，还很有商业头脑，这很酷。",
    "我想我可能会考虑两个方向：1. 成为一名专注于科技、媒体和电信（TMT）领域的律师。2. 加入一家法律科技公司的法务或产品部门。对于收入，我更看重长期的成长潜力和工作的创造性。如果和家人沟通，我会准备一些关于法律科技行业发展趋势的报告，以及一些新型法律岗位的薪酬数据来和他们讨论。",
    "我会选择 B 和 C。我计划先去联系那位做法律科技创业的学长，向他请教经验。同时，我会开始学习一些基础的Python编程知识，了解技术产品是如何开发的。"
]


# --- 2. 辅助函数和主逻辑 ---

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
        print("\n[可能的原因与解决方法]")
        print("1. API密钥或Endpoint错误: 'Authentication Fails' 或类似错误表明您的密钥或API地址不正确。")
        print("   -> 请检查您的 .env 文件中的 DEEPSEEK_API_KEY 是否正确。")
        print("   -> 确认API地址 'https://ark.cn-beijing.volces.com/api/v3' 是否为您的服务商提供的正确地址。")
        print("2. 网络问题: 检查您的网络连接是否可以访问火山方舟的API服务器。")
        print("3. 依赖库问题: 请确保您已安装 `langchain-openai` (pip install langchain-openai)。")
        print("─" * 80 + "\n")
        return None


def run_prototype(llm):
    """Runs the main interactive prototype."""
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
            history_messages = get_session_history("session").messages
            history_summary = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in history_messages])
            final_system_prompt = system_prompt.format(history=history_summary)
        else:
            final_system_prompt = system_prompt

        meta_prompt = f"""You are a thoughtful and insightful career planning coach... (rest of meta_prompt)"""
        prompt = ChatPromptTemplate.from_messages(
            [("system", meta_prompt), MessagesPlaceholder(variable_name="history"), ("human", "{input}")])
        chain = prompt | llm
        runnable_with_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="input",
                                                           history_messages_key="history")

        print("─" * 80)
        print(f"                                   {title}")
        print("─" * 80)

        if current_stage > 1:
            pass
        else:
            print_formatted(system_prompt, prefix="")

        print("\n💡 请针对以上问题，将你的想法和思考一次性输入，然后按Enter键。")
        user_input = input("你的回答: ")

        if user_input.lower() in ['quit', 'exit']:
            print_formatted("好的，对话结束。祝你一切顺利！")
            break

        print("\n[AI正在分析您的回答并提供建议，请稍候...]")
        response = runnable_with_history.invoke({"input": user_input},
                                                config={"configurable": {"session_id": "session"}})
        print_formatted(response.content)

        time.sleep(2)
        current_stage += 1

        if current_stage <= len(PROMPTS):
            print_formatted(PROMPTS[current_stage]["prompt"], prefix="")

    print("─" * 80)
    print("所有阶段已完成，感谢您的参与！")
    print("─" * 80)


def run_test_case(llm):
    """Runs an automated test case with predefined answers."""
    print("\n" + "=" * 30 + " 开始自动化测试 " + "=" * 30)

    # Use a copy of the test answers for the run
    test_answers_for_run = TEST_ANSWERS.copy()

    store = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    current_stage = 1
    while current_stage <= len(PROMPTS):
        if not test_answers_for_run:
            print("\n[测试错误] 测试答案不足，测试提前中止。")
            break

        stage_info = PROMPTS[current_stage]
        title = stage_info["title"]
        system_prompt = stage_info["prompt"]

        if current_stage == 4:
            history_messages = get_session_history("test_session").messages
            history_summary = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in history_messages])
            final_system_prompt = system_prompt.format(history=history_summary)
        else:
            final_system_prompt = system_prompt

        meta_prompt = f"""You are a thoughtful and insightful career planning coach... (rest of meta_prompt)"""
        prompt = ChatPromptTemplate.from_messages(
            [("system", meta_prompt), MessagesPlaceholder(variable_name="history"), ("human", "{input}")])
        chain = prompt | llm
        runnable_with_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="input",
                                                           history_messages_key="history")

        print("─" * 80)
        print(f"                                   {title}")
        print("─" * 80)

        if current_stage > 1:
            pass
        else:
            print_formatted(system_prompt, prefix="")

        # Simulate user input from the test data
        user_input = test_answers_for_run.pop(0)
        print("\n💡 预设的回答是:")
        print_formatted(user_input, prefix="模拟用户: ")

        print("\n[AI正在分析您的回答并提供建议，请稍候...]")
        response = runnable_with_history.invoke({"input": user_input},
                                                config={"configurable": {"session_id": "test_session"}})
        print_formatted(response.content)

        time.sleep(2)
        current_stage += 1

        if current_stage <= len(PROMPTS):
            print_formatted(PROMPTS[current_stage]["prompt"], prefix="")

    print("\n" + "=" * 30 + " 自动化测试结束 " + "=" * 30)


if __name__ == "__main__":
    print("AI职业规划助手原型启动 (使用火山方舟DeepSeek模型)...")

    llm_instance = get_llm_instance()

    if llm_instance:
        while True:
            choice = input("\n请选择运行模式: \n1. 交互模式\n2. 运行自动化测试用例\n\n请输入选项 (1或2): ")
            if choice == '1':
                run_prototype(llm_instance)
                break
            elif choice == '2':
                run_test_case(llm_instance)
                break
            else:
                print("无效输入，请输入 1 或 2。")

