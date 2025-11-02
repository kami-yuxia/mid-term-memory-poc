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