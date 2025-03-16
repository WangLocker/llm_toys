import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
import random
import numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_checkpoint = "/data01/wym/others/epoch_12_valid_blue_0.3163_model_weights.bin"
tokenizer_ckpt = "/data01/wym/models/t5-base-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(tokenizer_ckpt)
model.load_state_dict(torch.load(model_checkpoint))
model = model.to(device)

max_length = 384
stride = 128
answer_max_len = 12

model.eval()
while(True):
    ctx = [input("context: \n\n")]
    qus = [input("question: \n\n")]

    batch_data = tokenizer(
        qus,
        ctx,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        padding='max_length',
        return_tensors="pt",
        return_overflowing_tokens=True
    ).to(device)

    generated_tokens = model.generate(
            batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            max_length=answer_max_len,
            num_beams=3,
            no_repeat_ngram_size=2,
        ).cpu().numpy()

    if isinstance(generated_tokens, tuple):
        generated_tokens = generated_tokens[0]
    
    preds = ""
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    for pred in decoded_preds[0]:
        preds += pred.strip() 
    print("\n\n")
    print("answer:  \n\n")
    print(preds)
    print("\n\n")

#小米Max和小米Note 2系列是小米手机两个大屏幕手机系列,大屏手机很是受到一些网友的喜欢,小米Max是小米之前上线的手机,但现在已经下架了,但是不要着急,小米Max2即将发布了,小米Max2什么时候发布?米粉们有没有很期待,赶紧一起看看吧。|现在消息称小米Max2手机即将登场,根据爆料,这款手机会在本月也就是5月23日正式发布,而且新手机的宣传工作很快就会开启。|硬件方面,这款手机会采用6.4英寸屏幕,依然是1080P分辨率,电池容量提升到了5000mAh。有两个版本分别是骁龙626处理器+4GB RAM存储版本,以及骁龙660+6GB RAM存储版本。另外采用了F2.2光圈的1200万像素后置摄像头,前置摄像头像素为500万。|价格应该不会太贵预计从1699元起步,采用MIUI8操作系统,后置指纹识别并且具备金属机身设计。

#小米max2017开售时间
