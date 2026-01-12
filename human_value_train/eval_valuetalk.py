from typing import List, Any
import pandas as pd
import numpy as np

edge = 0.50 # 判断价值观为0还是1的阈值

input_file = "/home/zxy/for_sync/humanvalue/dataset/valuetalk_test_only_v10_ensemble.csv"
use_llm_or_bert = False

df = pd.read_csv(input_file)
df = df.fillna("")
# 只分析预测流程完成的
df = df[df['大模型汇总价值观']!=""]

# 去除可能的重复数据
df = df.drop_duplicates(subset="微博正文")
df = df.drop_duplicates(subset="id")
df = df.reset_index(drop=True)

# 不分析空文本
index_choose = list(df["微博正文"].apply(lambda x:len(x) > 1))
df = df.reset_index(drop=True)

# 将label的价值观特征翻译成标准的
label_translate = {
    "自我导向":'自主',
    "刺激":'刺激',
    "享乐主义":'享乐主义',
    "成就":'成就',
    "权力":'权力',
    "安全":'安全',
    "传统":'传统',
    "遵从":'遵从',
    "慈善":'友善',
    "大同主义":'博爱'
}
# 定义大模型价值观对应的数字值，只在测试大模型效果时有用
hanzi_value_dict = {
    "该价值观毫无体现": 0.0,
    "该价值观极微弱痕迹": 0.1,
    "该价值观轻微迹象": 0.3,
    "该价值观明确表达": 0.5,
    "该价值观显著表达": 0.8,
    "该价值观深刻贯彻": 1.0
}

# 确定原始label特征的顺序，valuetalk的标注和我们的不太一样
label_list_ori = [
    "自我导向",
    "刺激",
    "享乐主义",
    "成就",
    "权力",
    "安全",
    "传统",
    "遵从",
    "慈善",
    "大同主义"]
# 确定翻译后label特征的顺序，同时也是我们的特征顺序
label_list_trans = [
    '自主',
    '刺激',
    '享乐主义',
    '成就',
    '权力',
    '安全',
    '传统',
    '遵从',
    '友善',
    '博爱'
]

# 合并label中各个价值观的数据
label_flatten = []
for label_key in label_list_ori:
    label_flatten += df[label_key].tolist()

# 合并pred中各个价值观的数据
pred_flatten: list[Any] = []
for pred_key in label_list_trans:
    # 区分bert和大语言模型
    if use_llm_or_bert:
        # 如果是大语言模型，应该要从文字转化为数值
        pred_flatten += df['大模型汇总价值观'].apply(lambda x:float(hanzi_value_dict[eval(x)[pred_key]])).tolist()
    else:
        # bert则不需要
        pred_flatten += df['大模型汇总价值观'].apply(lambda x: float(eval(x)[pred_key])).tolist()

# 依据阈值转化为和valuetalk一样的二分类
pred_flatten = [abs(data) for data in pred_flatten]
pred_label = [1 if data > edge else 0 for data in pred_flatten]

label_flatten = np.array(label_flatten)
pred_flatten = np.array(pred_flatten)
pred_label = np.array(pred_label)

# 自己写的acc，由于验证计算是否正确
eval_result = [1 if label_flatten[i] == pred_label[i] else 0 for i in range(len(pred_label)) ]
eval_result = np.array(eval_result)
acc = eval_result.mean()
print(f"acc_my:{acc}")

# 换成标准的变量名
y = label_flatten
pred = pred_flatten
pred_label = pred_label


import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
print("AUC:",auc(fpr, tpr))

# 三维度分析数据

from sklearn.metrics import accuracy_score, recall_score, f1_score

acc = accuracy_score(abs(pred_label), abs(y))
print(f"acc:{acc}")
rec = recall_score(y, pred_label, average='micro')
print(f"rec:{rec}")
f1s = f1_score(y, pred_label, average='weighted')
print(f"f1_score:{f1s}")
from sklearn.metrics import classification_report

# 汇总报告
target_names = ['class 0', 'class 1']
print(classification_report(y, pred_label, target_names=target_names))







