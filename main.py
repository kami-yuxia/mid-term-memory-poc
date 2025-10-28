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
    state: MidTermMemoryChatState, llm: langchain_openai.ChatOpenAI
) -> MidTermMemoryChatState:
    # TODO: Implement a specific prompt for summarization
    summary_prompt = "Please summarize the following conversation history:\\n\\n"
    for msg in state["messages"]:
        role = (
            "User"
            if isinstance(msg, langchain_core.messages.HumanMessage)
            else "Assistant"
        )
        summary_prompt += f"{role}: {msg.content}\\n\\n"

    summary_prompt += (
        "\\nProvide a concise summary that captures the key points of the conversation."
    )

    # Create a temporary message to get the summary
    summary_messages = [
        langchain_core.messages.SystemMessage(
            content="You are a helpful assistant that summarizes conversations."
        ),
        langchain_core.messages.HumanMessage(content=summary_prompt),
    ]

    summary_response = llm.invoke(summary_messages)

    # Create a HumanMessage with is_compact flag in additional_kwargs
    new_messages: list[langchain_core.messages.AnyMessage] = [
        langchain_core.messages.HumanMessage(
            content=summary_response.content, additional_kwargs={"is_compact": True}
        )
    ]

    logger.info(
        f"\\033[93mConversation summarized: {summary_response.content[:100]}...\\033[0m"
    )  # Yellow

    session_id = state["session_id"]
    new_state: MidTermMemoryChatState = {
        "messages": new_messages,
        "session_id": session_id,
    }
    return new_state


def user_input(state: MidTermMemoryChatState) -> MidTermMemoryChatState:
    user_message = input("\nUser: ")
    session_id = state["session_id"]
    messages = state["messages"]
    messages.append(langchain_core.messages.HumanMessage(content=user_message))
    logger.info(f"\033[92mUser: {user_message}\033[0m")  # Green
    new_state: MidTermMemoryChatState = {"messages": messages, "session_id": session_id}
    return new_state


def should_continue(state: MidTermMemoryChatState) -> typing.Literal["continue", "end"]:
    user_decision = input("\nContinue? (y/n): ").strip().lower()
    if user_decision in ["y", "yes"]:
        return "continue"
    else:
        return "end"


def construct_workflow(
    db_path: pathlib.Path, llm: langchain_openai.ChatOpenAI, token_threshold: int = 1000
) -> langgraph.graph.state.CompiledStateGraph:
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
    workflow.add_node("summary_node", functools.partial(summary_node, llm=llm))
    workflow.add_node("should_continue", should_continue)

    workflow.set_entry_point("load_history")
    workflow.add_edge("load_history", "user_input")
    workflow.add_edge("user_input", "save_user_messages")
    workflow.add_edge("save_user_messages", "call_model")
    workflow.add_edge("call_model", "save_ai_messages")

    # Do we need a compaction now?
    workflow.add_conditional_edges(
        "save_ai_messages",
        functools.partial(should_summarize, token_threshold=token_threshold),
        {"summarize": "summary_node", "continue": "should_continue"},
    )

    workflow.add_edge("summary_node", "should_continue")

    # Does the user still want to continue to talk with us?
    workflow.add_conditional_edges(
        "should_continue",
        should_continue,
        {"continue": "user_input", "end": langgraph.graph.END},
    )

    app = workflow.compile()
    return app


if __name__ == "__main__":
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
    app = construct_workflow(db_path, llm)

    session_id = str(uuid.uuid4())
    initial_state = MidTermMemoryChatState(messages=[], session_id=session_id)

    logger.info("\033[95m=== Mid-Term Memory Chat Started ===\033[0m")  # Magenta
    logger.info(
        "\033[95mType your messages below. Press Ctrl+C to exit.\033[0m"
    )  # Magenta

    try:
        result = app.invoke(initial_state)
        logger.info("\033[95mChat completed.\033[0m")  # Magenta
    except KeyboardInterrupt:
        logger.info("\n\033[91mChat interrupted by user.\033[0m")  # Red
    except Exception as e:
        logger.error(f"An error occurred: {e}")
