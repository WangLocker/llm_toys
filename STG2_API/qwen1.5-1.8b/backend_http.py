from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union
import torch
import uvicorn
import time
from transformers import TextIteratorStreamer
from threading import Thread



@asynccontextmanager
async def lifespan(app:FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role:Literal['user','assistant','system']
    content:str

class ChatCompletionRequest(BaseModel):
    model:str
    messages:List[ChatMessage]
    temperature:Optional[float]=None
    top_p:Optional[float]=None
    max_length:Optional[int]=None
    stream:Optional[bool]=False

class ChatCompletionResponseChoice(BaseModel):
    index:int
    message:ChatMessage
    finish_reason:Literal['stop','length']

class DeltaMessage(BaseModel):
    role:Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None

class ChatCompletionResponseStreamChoice(BaseModel):
    index:int
    delta:DeltaMessage
    finish_reason:Optional[Literal["stop", "length"]]

class ChatResponseTokenInfo(BaseModel):
    prompt_tokens:int = None
    completion_tokens:int


class ChatCompletionResponse(BaseModel):
    model:str
    object:Literal["chat.completion", "chat.completion.chunk"]
    choices:List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time())) 
    usage:ChatResponseTokenInfo

@app.post("/v1/chat/completions",response_model=ChatCompletionResponse)
async def create_chat_completion(request:ChatCompletionRequest):
    global model,tokenizer,device
    if(request.messages[-1].role!='user'):
        raise HTTPException(status_code=400, detail="Invalid request")
    
    text = tokenizer.apply_chat_template(
        conversation=request.messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text],return_tensors="pt").to(device)

    if request.stream:
        generate = stream_predict_res(model_inputs,model_id=request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    length_of_first_generated = len(generated_ids[0])
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model,choices=[choice_data],usage=ChatResponseTokenInfo(completion_tokens=length_of_first_generated),object='chat.completion')

async def stream_predict_res(model_inputs,model_id:str):
    global model,tokenizer

    # 一个初始化块，并无实际内容
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )

    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    current_length = 0

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=1024)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        if len(new_text) == 0:
            continue
        current_length += len(new_text)
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))   

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'

if __name__ == "__main__":
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        "/data1/wym1/models/qwen_chat_1.8b",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("/data1/wym1/models/qwen_chat_1.8b")
    model.eval()

    uvicorn.run(app, host='0.0.0.0', port=12120, workers=1)


