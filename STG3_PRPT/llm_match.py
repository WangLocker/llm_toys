from rich import print
from rich.console import Console
from transformers import AutoTokenizer, AutoModel


# 提供所有类别以及每个类别下的样例
class_examples = {
        'sentence_pairs':[
            {
                'sentence1':'怎样更改个人资料图片？',
                'sentence2':'如何上传新的个人头像?',
                'label':'相似'
            },
            {
                'sentence1':'Python列表排序方法。',
                'sentence2':'Python中如何对列表进行升序排列。',
                'label':'相似'
            },
            {
                'sentence1':'北京热门旅游景点推荐。',
                'sentence2':'故宫博物院的历史文化价值。',
                'label':'不相似' 
            },
            {
                'sentence1':'iPhone14拍照功能评测。',
                'sentence2':'最新安卓手机摄像头性能对比。',
                'label':'不相似' 
            }
        ],
        'labels':['相似','不相似']
}

def init_prompts():
    """
    初始化前置prompt，便于模型做 incontext learning。
    """
    class_list = class_examples['labels']
    pre_history = [
        (
            f'现在你是一个文本匹配器，你需要按照要求判断我给你的句子对是否相似，根据结果输出{class_list}中的一个类别。',
            f'好的。'
        )
    ]

    for sentence_pair in class_examples['sentence_pairs']:
        sentence1 = sentence_pair['sentence1']
        sentence2 = sentence_pair['sentence2']
        _type = sentence_pair['label']
        pre_history.append((f'“{sentence1}”和 “{sentence2}” 是否相似？', _type))
    
    return {'class_list': class_list, 'pre_history': pre_history}

def inference(
        sentences: list,
        custom_settings: dict
    ):
    for sentence in sentences:
        with console.status("[bold bright_green] Model Inference..."):
            st1 = sentence[0]
            st2 = sentence[1]
            sentence_with_prompt = f'“{st1}”和 “{st2}” 是否相似？'
            response, history = model.chat(tokenizer, sentence_with_prompt, history=custom_settings['pre_history'])
        print(f'>>> [bold bright_red]sentence: {st1} , {st2}')
        print(f'>>> [bold bright_green]inference answer: {response}')
        # print(history)

if __name__ == '__main__':
    console = Console()
    tokenizer = AutoTokenizer.from_pretrained("D:\\jb\\models\\chatglm2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("D:\\jb\\models\\chatglm2-6b", trust_remote_code=True).cuda()
    model.eval()

    sentences = [
        ('如何修改头像', '可以通过上传图片修改头像吗'),
        ('王者荣耀司马懿连招', '王者荣耀司马懿有什么技巧'),
        ('王者荣耀司马懿连招', '历史上司马懿真的被诸葛亮空城计骗了吗'),
    ]
    
    custom_settings = init_prompts()
    inference(
        sentences,
        custom_settings
    )
