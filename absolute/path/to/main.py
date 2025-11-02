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


# 在现有代码基础上添加新的函数和修改主函数

def construct_simple_workflow(
    db_path: pathlib.Path,
    llm: langchain_openai.ChatOpenAI,
    summary_llm: langchain_openai.ChatOpenAI,
    token_threshold: int = 1000,
) -> langgraph.graph.state.CompiledStateGraph:
    """构造简化的线性工作流，无内部循环"""
    workflow = langgraph.graph.StateGraph(MidTermMemoryChatState)
    workflow.add_node("load_history", functools.partial(load_history, db_path=db_path))
    workflow.add_node("user_input", user_input)
    workflow.add_node("call_model", functools.partial(call_model, llm=llm))
    workflow.add_node("save_user_messages", functools.partial(save_messages, db_path=db_path))
    workflow.add_node("save_ai_messages", functools.partial(save_messages, db_path=db_path))
    workflow.add_node("save_compact_messages", functools.partial(save_messages, db_path=db_path))
    workflow.add_node("summary_node", functools.partial(summary_node, summary_llm=summary_llm))
    workflow.add_node("should_summarize", should_summarize)

    # 线性工作流结构
    workflow.set_entry_point("load_history")
    workflow.add_edge("load_history", "user_input")
    workflow.add_edge("user_input", "save_user_messages")
    workflow.add_edge("save_user_messages", "call_model")
    workflow.add_edge("call_model", "save_ai_messages")
    workflow.add_edge("save_ai_messages", "should_summarize")
    
    # 条件边：根据是否需要总结决定下一步
    workflow.add_conditional_edges(
        "should_summarize",
        functools.partial(should_summarize, token_threshold=token_threshold),
        {"summarize": "summary_node", "continue": "END"},  # 改为END结束工作流
    )
    
    # 总结节点之后的边
    workflow.add_edge("summary_node", "save_compact_messages")

    app = workflow.compile()
    return app


# 修改主函数
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
            try:
                # 每次都创建新的初始状态（但会从数据库加载历史）
                initial_state = MidTermMemoryChatState(messages=[], session_id=session_id)
                
                # 执行一轮对话（工作流内部不循环）
                result = app.invoke(initial_state, {"recursion_limit": 100})  # 正常递归限制
            except ExitConversationException:
                # 当用户输入结束指令时，退出循环
                break
    except KeyboardInterrupt:
        logger.info("\n\033[91mChat interrupted by user.\033[0m")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    logger.info("\033[95mChat completed.\033[0m")


# 在文件开头添加自定义异常类
class ExitConversationException(Exception):
    """当用户输入结束指令时抛出的异常"""
    pass


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