from typing import List, Any
import random
import pandas as pd
import numpy as np


edge = 0.5 # 判断价值观为0还是1的阈值
input_file = "/home/zxy/for_sync/humanvalue/dataset/valuetalk_test_only_v10_ensemble.csv"

# 用于提供参考聚类标签的文件
cluster_file = "/home/zxy/for_sync/humanvalue/dataset/valuetalk_test_only_v10_originalensemble_cluster.csv" # 采用测试集聚类结果，以统一不同结果的测量标准

# 读取结果文件
df = pd.read_csv(input_file)
df = df.fillna("")

# 读取测试集参考
df_cluster = pd.read_csv(cluster_file)
df_cluster = df_cluster.fillna("")

# 采用测试集的原始聚类结果，以统一不同预测结果的评测标准
if 'cluster_label' in df.columns:
    df = df.drop(columns="cluster_label")
# 合并文件
df = pd.merge(left=df, right=df_cluster[["id", "cluster_label"]], on="id", how="inner")
# 只分析预测流程完成的
df = df[df['大模型汇总价值观']!=""]

# 去除可能的重复数据
df = df.drop_duplicates(subset="微博正文")
df = df.drop_duplicates(subset="id")
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
hanzi_value_dict = {
    "该价值观毫无体现": 0.0,
    "该价值观极微弱痕迹": 0.1,
    "该价值观轻微迹象": 0.3,
    "该价值观明确表达": 0.5,
    "该价值观显著表达": 0.8,
    "该价值观深刻贯彻": 1.0
}
# 确定原始label特征的顺序,valuetalk的标注和我们的不太一样
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
def num_to_class(num):
    # 依据阈值，将数值转化为类别
    # 以群体为单位进行二分类，以适配valuetalk数据集本身的二分类标注
    if num < 0.5:
        return 0
    else:
        return 1

label_flatten = []
pred_flatten: list[Any] = []

# 去除聚类标签为空的
df = df[df["cluster_label"] != ""]
# 按照每个群体来评估，依据不同话题和不同聚类标签来区分
df["群体_name"] = df[["话题", "cluster_label"]].apply(lambda x:x[0] + str(int(x[1])), axis=1)

# 逐个群体进行评估
for i, event in enumerate(df['群体_name'].unique().tolist()):
    # 取出该群体的数据
    df_event = df[df['群体_name']==event]
    df_event = df_event.reset_index(drop=True)

    if len(df_event) < 5:
        # 该群体人太少了，分析不稳定，没法评测
        continue

    # 读取label
    for label_key in label_list_ori:
        # 统计群体占比
        label_flatten_i = df_event[label_key].mean()
        label_flatten_i = label_flatten_i
        # 转化标签
        label_flatten.append(num_to_class(label_flatten_i))


    # 读取pred
    for pred_key in label_list_trans:
        if False:
            # 用于评测大模型
            pred_flatten_i = df_event['大模型汇总价值观'].apply(lambda x:float(hanzi_value_dict[eval(x)[pred_key]])).tolist()
        else:
            # 用于评测小模型
            pred_flatten_i = df_event['大模型汇总价值观'].apply(lambda x: float(eval(x)[pred_key])).tolist()
        # 将多分类转化为和valuetalk一样的二分类
        pred_label_list = [1 if data >= edge else 0 for data in pred_flatten_i]
        # 统计群体占比
        pred_flatten_i = sum(pred_label_list)/len(pred_label_list)
        # 转化标签
        pred_flatten.append(num_to_class(pred_flatten_i))





label_flatten = np.array(label_flatten)
pred_flatten = np.array(pred_flatten)
pred_label = pred_flatten

# 自己写的acc，由于验证计算是否正确
eval_result = [1 if label_flatten[i] == pred_label[i] else 0 for i in range(len(pred_label)) ]
eval_result = np.array(eval_result)
acc = eval_result.mean()
print(f"acc_my:{acc}")

# 换成标准的变量名
y = label_flatten
pred = pred_flatten
pred_label = pred_label

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





