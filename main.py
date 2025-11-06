import functools
import os
import sqlite3
import sys
import pathlib
import typing
import uuid
import langchain_core.messages
import langchain_openai
import langgraph
import langgraph.graph
import langgraph.graph.state
from loguru import logger
from dotenv import load_dotenv

# 在文件开头添加自定义异常类
class ExitConversationException(Exception):
    """当用户输入结束指令时抛出的异常"""
    pass

def init_db(path: pathlib.Path) -> None:
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


class MidTermMemoryChatState(langgraph.graph.MessagesState):
    session_id: str


def load_history(
    state: MidTermMemoryChatState, db_path: pathlib.Path
) -> MidTermMemoryChatState:
    session_id = state["session_id"]
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        rows = cur.fetchall()

    loaded_messages: list[langchain_core.messages.AnyMessage] = []

    if rows:
        # Find the index of the last "compact" message, leave -1 when not found.
        last_compact_idx = -1
        for i in range(len(rows) - 1, -1, -1):  # Iterate backwards
            if rows[i][1] == "compact":  # [1] is the role
                last_compact_idx = i
                break

        start_idx = last_compact_idx if last_compact_idx != -1 else 0
        for i in range(start_idx, len(rows)):
            msg_id, role, content = rows[i][0], rows[i][1], rows[i][2]
            match role:
                case "user":
                    loaded_messages.append(
                        langchain_core.messages.HumanMessage(
                            content=content, additional_kwargs={"db_id": msg_id}
                        )
                    )
                case "assistant":
                    loaded_messages.append(
                        langchain_core.messages.AIMessage(
                            content=content, additional_kwargs={"db_id": msg_id}
                        )
                    )
                case "system":
                    loaded_messages.append(
                        langchain_core.messages.SystemMessage(
                            content=content, additional_kwargs={"db_id": msg_id}
                        )
                    )
                case "compact":
                    loaded_messages.append(
                        langchain_core.messages.HumanMessage(
                            content=content,
                            additional_kwargs={"db_id": msg_id, "is_compact": True},
                        )
                    )
                case _:
                    raise ValueError(f"Unknown message role: {role}")

    new_state: MidTermMemoryChatState = {
        "messages": loaded_messages,
        "session_id": session_id,
    }
    return new_state


def call_model(
    state: MidTermMemoryChatState, llm: langchain_openai.ChatOpenAI
) -> MidTermMemoryChatState:
    messages = state["messages"]
    session_id = state["session_id"]
    response = llm.invoke(messages)
    messages.append(response)

    logger.info(f"\033[96mAssistant: {response.content}\033[0m")  # Cyan

    new_state: MidTermMemoryChatState = {"messages": messages, "session_id": session_id}
    return new_state


def save_messages(
    state: MidTermMemoryChatState, db_path: pathlib.Path
) -> MidTermMemoryChatState:
    session_id = state["session_id"]
    messages = state["messages"]

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        cur.execute(
            "SELECT id FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
        )
        existing_db_ids = {row[0] for row in cur.fetchall()}

        for msg in messages:
            # Check if this message already exists in the database by looking at its db_id
            msg_db_id = msg.additional_kwargs.get("db_id")
            if msg_db_id is not None and msg_db_id in existing_db_ids:
                continue

            # Deal with compact message
            if isinstance(
                msg, langchain_core.messages.HumanMessage
            ) and msg.additional_kwargs.get("is_compact", False):
                role = "compact"
            elif isinstance(msg, langchain_core.messages.HumanMessage):
                role = "user"
            elif isinstance(msg, langchain_core.messages.AIMessage):
                role = "assistant"
            elif isinstance(msg, langchain_core.messages.SystemMessage):
                role = "system"
            else:
                raise ValueError(f"Unknown message type: {type(msg)}")

            cur.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (
                    session_id,
                    role,
                    msg.content,
                ),
            )
            
            # 获取新插入记录的ID并更新消息对象
            new_id = cur.lastrowid
            if new_id is not None:
                msg.additional_kwargs["db_id"] = new_id
                existing_db_ids.add(new_id)  # 添加到已存在ID集合中，防止同批次重复
                
        conn.commit()

    return state


def count_tokens(messages: list[langchain_core.messages.AnyMessage]) -> int:
    # Rough approximation: 1 token -> 4 characters for English text
    total_content = ""
    for msg in messages:
        total_content += str(msg.content)
    return len(total_content) // 4


def should_summarize(
    state: MidTermMemoryChatState, token_threshold: int = 1000
) -> typing.Literal["summarize", "continue"]:
    token_count = count_tokens(state["messages"])
    if token_count >= token_threshold:
        return "summarize"
    else:
        return "continue"


def summary_node(
    state: MidTermMemoryChatState, summary_llm: langchain_openai.ChatOpenAI
) -> MidTermMemoryChatState:
    messages = state["messages"]
    messages.extend(
        [
            langchain_core.messages.SystemMessage(
                content="You are a helpful AI assistant tasked with summarizing conversations."
            ),
            langchain_core.messages.HumanMessage(
                content="""\
Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits

- Errors that you ran into and how you fixed them
- Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.

2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests without confirming with the user first.
   If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:

   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:

   - [File Name 1]
     - [Summary of why this file is important]
     - [Summary of the changes made to this file, if any]
     - [Important Code Snippet]
   - [File Name 2]
     - [Important Code Snippet]
   - [...]

4. Errors and fixes:

   - [Detailed description of error 1]:
     - [How you fixed the error]
     - [User feedback on the error if any]
   - [...]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:

   - [Detailed non tool use user message]
   - [...]

7. Pending Tasks:

   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.

There may be additional summarization instructions provided in the included context. If so, remember to follow these instructions when creating the above summary. Examples of instructions include:
<example>

## Compact Instructions

When summarizing the conversation focus on typescript code changes and also remember the mistakes you made and how you fixed them.
</example>

<example>
# Summary instructions
When you are using compact - please focus on test output and code changes. Include file reads verbatim.
</example>
"""
            ),
        ]
    )

    summary_response = summary_llm.invoke(messages)

    # Create a HumanMessage with is_compact flag in additional_kwargs
    new_messages: list[langchain_core.messages.AnyMessage] = [
        langchain_core.messages.HumanMessage(
            content=summary_response.content, additional_kwargs={"is_compact": True}
        )
    ]

    logger.info(
        f"\033[93mConversation summarized: {summary_response.content[:100]}...\033[0m"
    )  # Yellow

    session_id = state["session_id"]
    new_state: MidTermMemoryChatState = {
        "messages": new_messages,
        "session_id": session_id,
    }
    return new_state


# 修改 user_input 函数
def user_input(state: MidTermMemoryChatState) -> MidTermMemoryChatState:
    user_message = input("\nUser: ")
    # 检查用户是否输入了结束指令
    if user_message.lower().strip() in ['n', 'no', '否']:
        raise ExitConversationException("User requested to exit conversation")
    
    session_id = state["session_id"]
    messages = state["messages"]
    messages.append(langchain_core.messages.HumanMessage(content=user_message))
    logger.info(f"\033[92mUser: {user_message}\033[0m")  # Green
    new_state: MidTermMemoryChatState = {"messages": messages, "session_id": session_id}
    return new_state

def construct_workflow(
    db_path: pathlib.Path,
    llm: langchain_openai.ChatOpenAI,
    summary_llm: langchain_openai.ChatOpenAI,
    token_threshold: int = 1000,
) -> langgraph.graph.state.CompiledStateGraph:
    from langgraph.graph import END  # 导入 END 常量
    
    workflow = langgraph.graph.StateGraph(MidTermMemoryChatState)
    workflow.add_node("load_history", functools.partial(load_history, db_path=db_path))
    workflow.add_node("user_input", user_input)
    workflow.add_node("call_model", functools.partial(call_model, llm=llm))
    # Use save_messages function for both nodes but with different names for workflow purposes
    workflow.add_node(
        "save_user_messages", functools.partial(save_messages, db_path=db_path)
    )
    workflow.add_node(
        "save_ai_messages", functools.partial(save_messages, db_path=db_path)
    )
    workflow.add_node(
        "save_compact_messages", functools.partial(save_messages, db_path=db_path)
    )
    workflow.add_node(
        "summary_node", functools.partial(summary_node, summary_llm=summary_llm)
    )
    # 注意：不再添加 end_node，直接使用 langgraph.graph.END

    workflow.set_entry_point("load_history")
    workflow.add_edge("load_history", "user_input")
    workflow.add_edge("user_input", "save_user_messages")
    workflow.add_edge("save_user_messages", "call_model")
    workflow.add_edge("call_model", "save_ai_messages")

    # 条件边：如果需要总结则去summary_node，否则去END
    workflow.add_conditional_edges(
        "save_ai_messages",
        functools.partial(should_summarize, token_threshold=token_threshold),
        {"summarize": "summary_node", "continue": END},  # 使用 END 而不是 "end_node"
    )

    workflow.add_edge("summary_node", "save_compact_messages")
    workflow.add_edge("save_compact_messages", END)  # 使用 END 而不是 "end_node"

    app = workflow.compile()
    return app

if __name__ == "__main__":
    # Initialize environment 
    load_dotenv()
    # Remove default logger
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO", colorize=True)

    db_path = pathlib.Path("./mid-term-memory-chat-history.db")
    init_db(db_path)

    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    llm = langchain_openai.ChatOpenAI(
        model=model_name, temperature=0.7, base_url=base_url
    )
    app = construct_workflow(db_path, llm, summary_llm=llm, token_threshold=500)

    session_id = os.getenv("SESSION_ID", str(uuid.uuid4()))
    initial_state = MidTermMemoryChatState(messages=[], session_id=session_id)

    logger.info("\033[95m=== Mid-Term Memory Chat Started ===\033[0m")  # Magenta
    logger.info(
        "\033[95mType your messages below. Press Ctrl+C to exit.\033[0m"
    )  # Magenta

    # 修改主函数中的执行部分
    try:
        while True:  # 外部循环控制多轮对话
            # 每次都创建新的初始状态（但会从数据库加载历史）
            initial_state = MidTermMemoryChatState(messages=[], session_id=session_id)
            
            # 执行一轮对话（工作流内部不循环）
            result = app.invoke(initial_state, {"recursion_limit": 100})  # 正常递归限制
            
    except ExitConversationException:
        logger.info("\033[95mChat completed.\033[0m") # Magenta
    except KeyboardInterrupt:
        logger.info("\n\033[91mChat interrupted by user.\033[0m") # Red
    except Exception as e:
        logger.error(f"An error occurred: {e}")