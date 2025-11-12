import os
import pathlib
import traceback
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# 加载环境变量
load_dotenv()

# 从 main.py 导入所需的函数和类，但不导入 construct_workflow
from main import (
    MidTermMemoryChatState,
    call_model,
    init_db,
    load_history,
    save_messages,
    summary_node,
    should_summarize
)


# 定义请求和响应模型
class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    response: str

# 设置最大 token 数量阈值
max_tokens=8000

# 初始化数据库
db_path = pathlib.Path("./mid-term-memory-chat-history.db")
init_db(db_path)

# 配置语言模型
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME"),
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    base_url=os.getenv("OPENAI_BASE_URL"),  # 添加 base_url
    api_key=os.getenv("OPENAI_API_KEY"),    # 显式添加 api_key
)


# 构建专门用于 API 的工作流程，不包含用户输入节点
def construct_api_workflow():
    from langgraph.graph import StateGraph, END
    from functools import partial

    workflow = StateGraph(MidTermMemoryChatState)

    # 添加节点
    workflow.add_node("load_history", partial(load_history, db_path=db_path))
    workflow.add_node("call_model", partial(call_model, llm=llm))
    workflow.add_node("save_messages", partial(save_messages, db_path=db_path))
    workflow.add_node("summary_node", partial(summary_node, summary_llm=llm))

    # 设置节点间的连接
    workflow.set_entry_point("load_history")
    workflow.add_edge("load_history", "call_model")
    workflow.add_edge("call_model", "save_messages")
    
    # 添加条件边，只有达到阈值才压缩
    workflow.add_conditional_edges(
        "save_messages",
        partial(should_summarize, token_threshold=max_tokens),
        {"summarize": "summary_node", "continue": END},
    )
    
    workflow.add_edge("summary_node", END)

    return workflow.compile()


# 创建 FastAPI 应用实例
app = FastAPI()

# 添加 CORS 中间件以允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该指定具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # 构建工作流程
        app_workflow = construct_api_workflow()

        # 准备输入数据
        input_data = {
            "messages": [HumanMessage(content=request.message)],
            "session_id": request.session_id,
        }

        # 执行工作流程
        result = await app_workflow.ainvoke(input_data)
        
        # 打印调试信息
        print(f"Result keys: {result.keys()}")
        print(f"Messages: {result['messages']}")
        
        # 检查结果中是否有消息
        if "messages" not in result or not result["messages"]:
            raise HTTPException(status_code=500, detail="No messages returned from workflow")

        # 查找最后一条 AI 消息
        last_ai_message = None
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage):
                last_ai_message = message
                break
        
        # 如果找到了 AI 消息，返回它
        if last_ai_message:
            return ChatResponse(response=last_ai_message.content)
        else:
            # 如果没有找到 AI 消息，返回最后一条消息
            last_message = result["messages"][-1]
            return ChatResponse(response=f"Received {type(last_message).__name__}: {last_message.content}")

    except Exception as e:
        # 记录详细的错误信息
        error_details = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_details)  # 打印到控制台以便调试
        raise HTTPException(status_code=500, detail=error_details)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)