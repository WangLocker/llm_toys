模型采用Google T5-Base。
问答模型是指通过输入一个问题和一段文章，输出问题的答案。
问答模型分为抽取式和生成式，我们本次将使用生成式模型来训练一个问答模型。
我们选用T5作为 backbone，使用提供的数据集来训练得到一个生成式的问答模型。
数据的格式如下：
{"context": "违规分为:一般违规扣分、严重违规扣分、出售假冒商品违规扣分,淘宝网每年12月31日24:00点会对符合条件的扣分做清零处理,详情如下:|温馨提醒:由于出售假冒商品24≤N<48分,当年的24分不清零,所以会存在第一年和第二年的不同计分情况。", "answer": "12月31日24:00", "question": "淘宝扣分什么时候清零", "id": 203}
每一行为一个数据样本，json 格式。
其中，"context" 代表参考文章，question 代表问题，"answer" 代表问题答案。
模型的评价指标采用BLEU-1，BLEU-2，BLEU-3，BLEU-4。