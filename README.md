# 价值观分析和训练
正在更新中
## 模型概述
基于Debert的模型，搭配llm，robert等进行集成分析，并具备帖子过滤和多模态增强功能

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

