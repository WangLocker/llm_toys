from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler

checkpoint = '/data01/wym/models/t5-base-chinese-cluecorpussmall'
train_dir = '/data01/wym/datas/train.json'
valid_dir = '/data01/wym/datas/dev.json'

# 数据加载
class QADataset(Dataset):

    def __init__(self,data_dir):
        super().__init__()
        self.data = self.load_data(data_dir)

    def load_data(self, data_dir):
        Data = {}
        with open(data_dir,'rt',encoding='utf-8') as f:
            for idx,line in enumerate(f):
                json_obj = json.loads(line)
                Data[idx] = json_obj
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

train_data = QADataset(train_dir)
valid_data = QADataset(valid_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using device{device}")


#数据预处理
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
model = model.to(device)

max_length = 384
stride = 128
answer_max_len = 12

def collote_fn(batch_samples):
    batch_inputs_ctx, batch_inputs_que, batch_targets = [],[],[]
    for sample in batch_samples:
        batch_inputs_ctx.append(sample['context'])
        batch_inputs_que.append(sample['question'])
        batch_targets.append(sample['answer'])

    batch_data = tokenizer(
        batch_inputs_que,
        batch_inputs_ctx,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        padding='max_length',
        return_tensors="pt",
        return_overflowing_tokens=True
    )

    sample_mapping = batch_data.pop('overflow_to_sample_mapping')

    labels = tokenizer(
        batch_targets, 
        padding=True, 
        truncation = True,
        max_length=answer_max_len,
        return_tensors="pt"
    )['input_ids'].clone()

    labels_overflow = labels[sample_mapping]
    end_token_index = torch.where(labels_overflow == tokenizer.sep_token_id)[1]
    for idx, end_idx in enumerate(end_token_index):
        labels_overflow[idx][end_idx+1:] = -100
    batch_data['labels'] = labels_overflow
    
    return batch_data

train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=8, shuffle=False, collate_fn=collote_fn)

from tqdm.auto import tqdm

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss, loss_record):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss:{0:>7f}')
    finish_batch_num = (epoch - 1) * (len(dataloader))

    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data.pop("token_type_ids", None)
        batch_data = batch_data.to(device)
        outputs = model(**batch_data)
        loss = outputs.loss
        loss_record.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)

    return total_loss

learning_rate = 5e-5
epoch_num = 20

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

import nltk
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

def test_loop(dataloader, model):
    preds, labels = [], []
    scores = []
    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=answer_max_len,
                num_beams=4,
                no_repeat_ngram_size=2,
            ).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        label_tokens = batch_data["labels"].cpu().numpy()

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [' '.join(pred.strip()) for pred in decoded_preds]
        labels += [' '.join(label.strip()) for label in decoded_labels]

    for pred, label in zip(preds, labels):
    # 将标签和预测都转为列表格式，句子的 BLEU 分数是基于多个参考
        score = sentence_bleu([label.split()], pred.split(), weights=(0.15, 0.35, 0.35, 0.15))
        scores.append(score)
    avg_result = np.mean(scores)
    return avg_result



total_loss = 0.
best_avg_score = 0.
loss_record = []
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    loss_record_this_e = []
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss, loss_record_this_e)
    loss_record.append(np.mean(loss_record_this_e))
    avg_blue_score = test_loop(valid_dataloader, model)
    print(f"Valid Score (Avg) : {avg_blue_score}\n-------------------------------")
    if avg_blue_score > best_avg_score:
        best_avg_score = avg_blue_score
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'/data01/wym/others/epoch_{t+1}_valid_blue_{avg_blue_score:0.4f}_model_weights.bin')
print("Done!")

import matplotlib.pyplot as plt

# 画图
plt.figure(figsize=(8, 5))
plt.plot(loss_record, marker='o', linestyle='-', markersize=5, label="Training Loss")

# 添加标题和标签
plt.title("Loss Convergence Curve", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.grid(True)

# 保存图片
plt.savefig("/home/wym/llm_demos/STG1_T5QA/loss_curve.png", dpi=300)

# 显示图像
plt.show()
