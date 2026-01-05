# 价值观分析和训练
正在更新版本中
## 模型概述
本模型是一个基于bert搭配多模态大语言模型MLLM的微博价值观分析模型。模型采用关键词和LLM筛选，来去除噪声和无价值观数据，并可使用MLLM增加图片信息。模型本体通过集成bert模型实现对微博的施瓦茨价值观分析，并通过聚类来可视化群体价值观分布。
本模型下附属统计模块，负责统计日帖子数，词云图等信息，用以补充

## 核心功能
### 算法输入
输入为微博帖子数据所在文件夹，帖子依据所属事件分开整理，并可附有图像文件夹

### 算法输出
输出为个体和群体分析的汇总dict数据，可直接提交给服务器

### 输出详解
个体结果：帖子所表达的各施瓦茨价值观（'博爱', '友善', '权力', '成就', '传统', '遵从', '安全', '自主', '刺激', '享乐主义'）强度，划分和对应参考数值如下：
  "该价值观毫无体现": 0.0,
  "该价值观极微弱痕迹": 0.1,
  "该价值观轻微迹象": 0.3,
  "该价值观明确表达": 0.5,
  "该价值观显著表达": 0.8,
  "该价值观深刻贯彻": 1.0,
群体结果：
  群体组成：依据价值观聚类后该事件分为几类群体
  典型群体价值观构成：该事件每类群体的平均价值观强度
  价值观突出帖子列表：价值观最为尖锐突出的帖子及其价值观列表

### 技术亮点与分析流程详解
  0. 统一调用接口（humanvalue_predict/predict_human_value_api.py）初始化模型，并依次执行以下流程
  1. 关键词过滤（humanvalue_predict/filter_datasets.py）读取由价值观原文，相关词，相近词及微博语境相近词构成的关键词筛选列表，并对数据进行初筛，筛选率约50%
  2. 多模态增强（humanvalue_predict/get_multi_pic_caption.py）使用多模态大模型描述图片内容，以提供额外的多模态信息
  3. LLM筛选与增强（humanvalue_predict/LLM_filter_datasets_qwen.py）使用大模型依据语义去除广告，白开水，事实陈述等无用信息，并进行概括和逐句翻译，筛选率约50%
  4. 集成分析（humanvalue_predict/bert_content_predict.py）使用竞赛第一的英文DeBert和RoBert模型，分别进行整体分析和逐句分析，并汇总
  5. bert汇总（humanvalue_predict/bert_ensemble_cyx.py）使用训练好的中文DeBert模型，基于原文，用户，帖子属性和bert分析结果，给出最终的价值观分析结果
  6. 群体分析（humanvalue_predict/k_cluster.py）基于分析好的个体结果，使用聚类算法分析群体构成和典型群体价值观，并整理所有分析结果，转化为提交服务器的格式

  1. 统计分析接口（humanvalue_predict/static_analyize_api.py）统计帖子随时间变化趋势，地域分布，转赞评数量，并绘制词云图

## 价值观分析服务流程
### 下载模型
链接：https://pan.quark.cn/s/d7f8877ddbc8，存放于/humanvalue/human_value_predict/humanvalue_api/checkpoints/下
#### 分析指令
```bash
cd human_value_predict
python predict_human_value_api.py
# 先init()进行初始化
# 然后forward()进行分析，输入event_data_csv_path为目标文件所在文件夹地址
# 返回为分析结果dict
```
### 输出
可以直接提交给服务器的dict
中间数据会保存在human_value_data下，以便复用

### 参数设置
位于human_value_predict下basic_config.py处

## 训练流程

### 数据准备
**数据源：**
- 下载数据，放在dataset下，并设置train.py中的数据名称
- 训练数据：基础数据training_data_with_weights.parquet或包含武汉的扩充数据wuhan_train_weighted.parquet
- 验证数据集：由训练数据集切分得到
数据存放地址：
夸克网盘：https://pan.quark.cn/s/924335bdb441

### 训练

#### 训练指令
```bash
cd human_value_train
python train.py
```

#### 输出
- 模型默认保存至 `./checkpoints`目录

