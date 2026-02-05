# 价值观分析和训练
正在更新版本中
## 模型概述
本模型是一个基于bert搭配多模态大语言模型MLLM的微博价值观分析模型。模型采用关键词和LLM筛选，来去除噪声和无价值观数据，并可使用MLLM增加图片信息。模型本体通过集成bert模型实现对微博的施瓦茨价值观分析，并通过聚类来可视化群体价值观分布。
本模型下附属统计模块，负责统计日帖子数，词云图等信息，用以补充
相关模型:
cyx_model，基于https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese进一步训练的模型，用于最终的集成预测。
  相关论文：Fengshenbang 1.0: Being the Foundation of Chinese Cognitive Intelligence，感谢他们的工作和很棒的预训练模型
bert_model，包含https://huggingface.co/danschr/roberta-large-BS_16-EPOCHS_5-LR_5e-05-ACC_GRAD_2-MAX_LENGTH_165/tree/main，和https://huggingface.co/tum-nlp/Deberta_Human_Value_Detector。
  该模型是作为价值观比赛获奖方案的一部分，用于多角度分析，相关论文：Adam-Smith at SemEval-2023 Task 4: Discovering Human Values in Arguments with Ensembles of Transformer-based Models，感谢他们的工作
qwen_model，基于API的大模型，用于进行图片描述，文本总结和文本翻译，在此处我们采用QWEN2.5VL，亦可尝试其他模型。
  其技术报告为Qwen2.5-VL Technical Report，感谢他们的工作和好用的大模型

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
### 环境配置
分析服务的环境于humanvalue/humanvalue/human_value_predict/requirements.txt下，但可能缺乏些服务器上的环境，请以服务器上的配置为准
训练环境主要依赖为在python 3.11.14下，torch==2.9.1，lightning==2.6.0，transformers==4.55.0，pandas==2.3.3，scikit-learn==1.8.0, typing，具体库版本或许可以调整



## 价值观分析服务流程
### 下载模型
链接：https://pan.quark.cn/s/c996b6d0e94f，存放于/humanvalue/human_value_predict/humanvalue_api/checkpoints/下
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
- 训练数据：基础数据train_vt_only_v10_weighted.parquet或包含武汉的扩充数据wuhan_train_vt_only_v10_weighted.parquet
- 验证数据集：使用valuetalk_val_only_v10.csv，是我们切分出来的只有200条数据的验证集
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

### 推理
- 使用专为valuetalk设计的human_value_train/bert_predict.py，设置好模型位置后运行即可

## 测试
- 个体测试通过human_value_train/eval_valuetalk.py，设置好推理所得文件位置后运行即可
- 群体测试通过human_value_train/eval_valuetalk_event.py，设置好推理所得文件位置后运行即可

# 如何增加新的数据集
- 1. 将目标数据集经过human_value_predict后得到符合训练的格式的数据
- 2. 对其进行标注，比如通过大模型等
- 3. 使用human_value_train/data/data_preprocess处理其所在文件夹，将数据文件与已有数据进行合并
- 4. 使用human_value_train/analyiz_value_realif_simple.py为数据文件生成数据权重
- 5. 进行训练

## 推理服务参数详解
###关键词筛选部分
ord_file，用于指定本地的价值观关键词原始词表位置
weibo_file，用于指定本地的价值观关键词微博化词表位置

### 大语言模型筛选和增强部分
api_key，用于指定大模型的API_KEY
api_url，用于指定大模型访问的远程端口或本地端口

use_parallel_if，是否并发请求大模型
parallel_max_num，并发请求最大数量
batch，多少个数据打包成一批一起处理，需注意，对于翻译来说，其实际为batch*平均句子数
use_local_pic_if，是否使用本地图片，如果为ture，将读取pic下文件，如果为false，将尝试将图片url传递给大模型

content_model，所使用的文本大模型名称
multi_model，所使用的多模态(图文)大模型
multi_modal，是否使用在文本处理时使用多模态

### bert集成分析部分

model_config_dic，所使用的bert模型配置表
checkpoint_path，所训练模型ckpt文件位置
model_name，模型的hugging face名称，用于在本地没有时，联网获取
local_model_path，模型的本地存放位置，用于读取模型及tokenizer
label_columns，模型的输出类别名称和顺序

cyx_model，所使用的训练好的模型的配置
bert_model_list，所使用的多个bert模型的配置
llm_model，所使用的大模型的接口配置

### k均值聚类和可视化部分
enhance_performence，是否对结果缩放以获得更好的视觉效果，由于采用雷达图，如果有价值观为0，会造成雷达图上的尖刺感，不好看，由此对数据进行了缩放
llm_or_bert，使用llm格式（我们预训练好的模型也是这个）还是原始bert格式（指的是bert_model_list中的未训练模型输出格式）
sharp_internel_weight，选择尖锐价值观样本时的权重
one_file_with_multi_event，一个文件中是否可能有多个文件，如果为是，将对该文件内的每个事件分别处理
trans_data_screen，最后的输出格式接口，方便调整
