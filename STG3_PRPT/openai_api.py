# coding=utf-8
# Implements API for ChatGLM2-6B in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.


# 导入必要的库
import time
import torch
import uvicorn
from pydantic import BaseModel, Field  # 用于数据验证和序列化
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware  # 处理跨域请求
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union
from transformers import AutoTokenizer, AutoModel  # 用于加载模型和分词器
from sse_starlette.sse import ServerSentEvent, EventSourceResponse  # 用于流式响应

# 应用生命周期管理器，用于清理GPU内存
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# 创建FastAPI应用实例
app = FastAPI(lifespan=lifespan)

# 配置CORS中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义数据模型
class ModelCard(BaseModel):
    """模型信息卡片，用于描述模型基本信息"""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None

class ModelList(BaseModel):
    """模型列表，包含多个ModelCard"""
    object: str = "list"
    data: List[ModelCard] = []

class ChatMessage(BaseModel):
    """聊天消息格式"""
    role: Literal["user", "assistant", "system"]  # 消息角色：用户、助手或系统
    content: str  # 消息内容

class DeltaMessage(BaseModel):
    """用于流式响应的增量消息格式"""
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    """聊天补全请求的格式
    用于接收客户端发送的对话请求，包含了进行对话所需的所有参数
    遵循 OpenAI API 格式规范
    """
    model: str  # 使用的模型名称
    messages: List[ChatMessage]  # 对话历史
    temperature: Optional[float] = None  # 温度参数，控制随机性
    top_p: Optional[float] = None  # 核采样参数
    max_length: Optional[int] = None  # 最大生成长度
    stream: Optional[bool] = False  # 是否使用流式响应

class ChatCompletionResponseChoice(BaseModel):
    """非流式响应的选项格式
    用于封装模型的单次完整响应内容
    当 stream=False 时使用此格式
    """
    index: int  # 响应的序号，用于多选项场景
    message: ChatMessage  # 完整的响应消息
    finish_reason: Literal["stop", "length"]  # 响应结束的原因：自然结束或达到长度限制

class ChatCompletionResponseStreamChoice(BaseModel):
    """流式响应的选项格式
    用于封装模型的增量响应内容
    当 stream=True 时使用此格式
    """
    index: int  # 响应的序号
    delta: DeltaMessage  # 增量的响应内容
    finish_reason: Optional[Literal["stop", "length"]]  # 响应结束的原因，流式传输中可能为空

class ChatCompletionResponse(BaseModel):
    """聊天补全响应的总体格式
    包装最终返回给客户端的完整响应数据
    可以包含流式或非流式的响应内容
    """
    model: str  # 使用的模型名称
    object: Literal["chat.completion", "chat.completion.chunk"]  # 响应类型：完整响应或增量响应
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]  # 响应内容列表
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))  # 响应创建时间

# API端点实现
@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """返回可用模型列表的端点"""
    global model_args
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """处理聊天请求的主要端点
    
    这是处理聊天请求的核心函数，负责：
    1. 处理用户输入的消息
    2. 管理对话历史
    3. 调用模型生成响应
    4. 支持流式和非流式两种响应模式
    
    参数:
        request (ChatCompletionRequest): 包含用户请求的所有信息，包括：
            - 模型名称
            - 对话历史
            - 温度等生成参数
            - 是否使用流式响应
    
    返回:
        ChatCompletionResponse: 模型的响应数据
            - 非流式模式：返回完整的响应内容
            - 流式模式：返回 EventSourceResponse 对象，用于流式传输
    
    异常:
        HTTPException: 当请求格式不正确时抛出 400 错误
    """
    global model, tokenizer

    # 验证请求的最后一条消息是否来自用户
    # OpenAI API 要求最后一条消息必须是用户消息
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    
    # 提取用户的最新查询内容
    query = request.messages[-1].content

    # 处理系统提示和历史消息
    # 如果存在系统提示，将其与用户查询合并
    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        query = prev_messages.pop(0).content + query

    # 构建对话历史
    # 将历史消息按照 user-assistant 对的形式组织
    # 确保历史消息是成对的（用户-助手）
    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i].role == "user" and prev_messages[i+1].role == "assistant":
                history.append([prev_messages[i].content, prev_messages[i+1].content])

    # 处理流式响应请求
    # 如果客户端请求流式响应，调用 predict 函数逐步生成内容
    if request.stream:
        generate = predict(query, history, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    # 非流式响应处理
    # 等待模型生成完整响应后一次性返回
    response, _ = model.chat(tokenizer, query, history=history)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    # 返回完整的响应对象
    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")

async def predict(query: str, history: List[List[str]], model_id: str):
    """流式响应生成器函数
    
    实现流式输出（打字机效果）的核心函数，主要功能：
    1. 初始化助手角色
    2. 逐步生成并返回模型响应
    3. 处理响应结束信号
    
    参数:
        query (str): 用户的当前问题
        history (List[List[str]]): 对话历史记录，格式为 [[用户问题1, 助手回答1], ...]
        model_id (str): 使用的模型标识符
    
    生成器返回值:
        str: JSON格式的响应数据块，包括：
            - 初始块：设置助手角色
            - 内容块：增量的回复内容
            - 结束块：标记响应结束
            - 最后返回 [DONE] 表示流式传输完成
    """
    global model, tokenizer

    # 第一步：初始化助手角色
    # 创建初始响应块，设置角色为 assistant
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    # 用于追踪已生成的文本长度
    current_length = 0

    # 第二步：流式生成回复内容
    # 通过 stream_chat 方法获取模型的增量输出
    for new_response, _ in model.stream_chat(tokenizer, query, history):
        # 跳过重复的内容
        if len(new_response) == current_length:
            continue

        # 计算新生成的文本内容
        new_text = new_response[current_length:]
        current_length = len(new_response)

        # 创建包含新文本的响应块
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    # 第三步：发送结束信号
    # 创建表示生成结束的响应块
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    # 发送最终的结束标记
    yield '[DONE]'

# 主程序入口
if __name__ == "__main__":
    # 检查显卡是否可用
    if torch.cuda.is_available():
        print("当前设备的显卡可用。")
        # 获取可用的 GPU 数量
        num_gpus = torch.cuda.device_count()
        print(f"可用的 GPU 数量为: {num_gpus}")
        # 获取当前使用的 GPU 设备索引
        current_device = torch.cuda.current_device()
        print(f"当前使用的 GPU 设备索引为: {current_device}")
        # 获取当前 GPU 设备的名称
        device_name = torch.cuda.get_device_name(current_device)
        print(f"当前 GPU 设备的名称为: {device_name}")
    else:
        print("当前设备的显卡不可用。")

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("D:\jb\models\chatglm2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("D:\jb\models\chatglm2-6b", trust_remote_code=True)
    # 检查是否有可用的 GPU 并将模型移动到 GPU
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # 检查模型是否在 GPU 上
    if next(model.parameters()).is_cuda:
        print("模型已成功加载到 GPU 上。")
    else:
        print("模型未加载到 GPU 上，可能在 CPU 上运行。")

    # 启动服务器
    uvicorn.run(app, host='0.0.0.0', port=12120, workers=1)
