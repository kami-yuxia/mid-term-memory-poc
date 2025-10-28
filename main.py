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
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        rows = cur.fetchall()

    loaded_messages: list[langchain_core.messages.AnyMessage] = []
    for role, content in rows:
        match role:
            case "user":
                loaded_messages.append(
                    langchain_core.messages.HumanMessage(content=content)
                )
            case "assistant":
                loaded_messages.append(
                    langchain_core.messages.AIMessage(content=content)
                )
            case "system":
                loaded_messages.append(
                    langchain_core.messages.SystemMessage(content=content)
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
        cur.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        existing_count = cur.fetchone()[0]

        # If message count in state is greater than in DB, save the new ones
        if len(messages) > existing_count:
            new_messages = messages[existing_count:]
            for msg in new_messages:
                if isinstance(msg, langchain_core.messages.HumanMessage):
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
    db_path: pathlib.Path, llm: langchain_openai.ChatOpenAI
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
    workflow.add_node("should_continue", should_continue)

    workflow.set_entry_point("load_history")
    workflow.add_edge("load_history", "user_input")
    workflow.add_edge("user_input", "save_user_messages")
    workflow.add_edge("save_user_messages", "call_model")
    workflow.add_edge("call_model", "save_ai_messages")
    workflow.add_conditional_edges(
        "save_ai_messages",
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
