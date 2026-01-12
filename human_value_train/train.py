import os
import sys

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
sys.path.append(current_dir)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'

from lightning import Trainer, seed_everything
RANDOM_SEED = 1 # 设置随机种子

seed_everything(RANDOM_SEED)
import random
import pandas as pd
import torch
import numpy as np

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch.nn as nn
from data.DataModule import BertDataModule
from model.IntergrateBERT import DebertaMultiLabelClassifier
import warnings
warnings.filterwarnings('ignore')

PARAMS = {

    "USE_COSINE_SCHEDULER": True,  # 启用余弦学习率调度
    "COSINE_SCHEDULER_TYPE": "cosine_annealing",  # 可选: "cosine_annealing" 或 "cosine_warm_restarts"
    "MIN_LEARNING_RATE": 1e-7,  # 最小学习率
    "WARMUP_STEPS_RATIO": 0.1,  # warmup步骤比例
    "COSINE_T_MULT": 2,  # 用于cosine_warm_restarts，周期倍增因子
    "COSINE_RESTART_PERIOD": 1,  # 用于cosine_warm_restarts，初始周期（epoch数）
    "NUM_CYCLES": 2,  # 2次重启（共3个周期）
    "TRAIN_PATH": "./dataset/train_vt_only_v10_weighted.parquet",
    # Language Model and Hyperparameters
    "MODEL_NAME": 'IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese',
    # "BATCH_SIZE": 16,
    "BATCH_SIZE": 16,
    "ACCUMULATE_GRAD_BATCHES": 1,
    "LEARNING_RATE": 1e-4,
    "EPOCHS": 5,
    "OPTIMIZER": 'AdamW',
    "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "NUM_LABELS": 10,
    "NUM_CLASSES_PER_LABEL": 6,
    "NUM_TRAIN_WORKERS": 0,
    "NUM_VAL_WORKERS": 0,
    "MAX_TOKEN_COUNT": 512,
    "RANDOM_SEED": RANDOM_SEED,  # Random Seed Selected for this Training Run

    # Additional Dropout or Additional Hidden Layers (Not used for the final submission)
    "DROPOUT": 0.2,  # e.g 0.5 (float)
    "HIDDEN_LAYERS": [(128, nn.ReLU())],  # 没有武汉数据的则采用128即可，武汉数据用256
    "WEIGHT_DECAY": 0.1,
    "CRITERION": [nn.CrossEntropyLoss()],

    # Early Stopping Params
    "PATIENCE": 300,
    "VAL_CHECK_INTERVAL": 176,

    # The metric we optimize for. Alternative "custom_f1/Val" and "max"
    "MAX_THRESHOLD_METRIC": "custom",  # The f1-score that should maximized (custom = formula for the task evaluation)
    "EARLY_STOPPING_METRIC": "val_overall_accuracy",
    "EARLY_STOPPING_MODE": "max",

    # save dir
    "SAVE_DIR":"/data1/bert/humanvalue/",

    # DATA
    "VALIDATION_SET_SIZE": 100,
    "LABEL_NAMES": [
        "博爱", "友善", "权力", "成就", "传统",
        "遵从", "安全", "自主", "刺激", "享乐主义"
    ],
}

print(PARAMS)


def set_seed_everything(seed=42):
    """设置所有随机种子以确保可复现性"""
    # 1. Lightning 的种子设置
    seed_everything(seed, workers=True)

    # 2. PyTorch 相关
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时

    # 3. CUDA 确定性设置（影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 4. Python 和 numpy
    np.random.seed(seed)
    random.seed(seed)

    # 5. 设置环境变量（如果使用子进程）
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # 6. 启用PyTorch的确定性算法（某些操作可能不支持）
    torch.use_deterministic_algorithms(True, warn_only=True)


# 在训练开始前调用
set_seed_everything(PARAMS["RANDOM_SEED"])











# 读入训练集
train_df = pd.read_parquet(PARAMS["TRAIN_PATH"])

test_columns = ['text', '博爱', '友善', '权力', '成就', '传统', '遵从', '安全', '自主', '刺激', '享乐主义']
# 读入valuktalk验证集，以测试其效果
val_df = pd.read_csv("/home/zxy/for_sync/humanvalue/dataset/valuetalk_val_only_v10.csv", encoding="utf_8_sig")
# 修正验证集的格式
if "博爱" not in val_df.columns and "大同主义" in val_df.columns:
    val_df["博爱"] = val_df["大同主义"]
if "友善" not in val_df.columns and "慈善" in val_df.columns:
    val_df["友善"] = val_df["慈善"]
if "自主" not in val_df.columns and "自我导向" in val_df.columns:
    val_df["自主"] = val_df["自我导向"]
val_df = val_df[test_columns]
print(len(val_df))
# 确定val数据的label维度
LABEL_COLUMNS = ['博爱_该价值观毫无体现', '博爱_该价值观极微弱痕迹', '博爱_该价值观轻微迹象', '博爱_该价值观明确表达',
       '博爱_该价值观显著表达', '博爱_该价值观深刻贯彻', '友善_该价值观毫无体现', '友善_该价值观极微弱痕迹',
       '友善_该价值观轻微迹象', '友善_该价值观明确表达', '友善_该价值观显著表达', '友善_该价值观深刻贯彻',
       '权力_该价值观毫无体现', '权力_该价值观极微弱痕迹', '权力_该价值观轻微迹象', '权力_该价值观明确表达',
       '权力_该价值观显著表达', '权力_该价值观深刻贯彻', '成就_该价值观毫无体现', '成就_该价值观极微弱痕迹',
       '成就_该价值观轻微迹象', '成就_该价值观明确表达', '成就_该价值观显著表达', '成就_该价值观深刻贯彻',
       '传统_该价值观毫无体现', '传统_该价值观极微弱痕迹', '传统_该价值观轻微迹象', '传统_该价值观明确表达',
       '传统_该价值观显著表达', '传统_该价值观深刻贯彻', '遵从_该价值观毫无体现', '遵从_该价值观极微弱痕迹',
       '遵从_该价值观轻微迹象', '遵从_该价值观明确表达', '遵从_该价值观显著表达', '遵从_该价值观深刻贯彻',
       '安全_该价值观毫无体现', '安全_该价值观极微弱痕迹', '安全_该价值观轻微迹象', '安全_该价值观明确表达',
       '安全_该价值观显著表达', '安全_该价值观深刻贯彻', '自主_该价值观毫无体现', '自主_该价值观极微弱痕迹',
       '自主_该价值观轻微迹象', '自主_该价值观明确表达', '自主_该价值观显著表达', '自主_该价值观深刻贯彻',
       '刺激_该价值观毫无体现', '刺激_该价值观极微弱痕迹', '刺激_该价值观轻微迹象', '刺激_该价值观明确表达',
       '刺激_该价值观显著表达', '刺激_该价值观深刻贯彻', '享乐主义_该价值观毫无体现', '享乐主义_该价值观极微弱痕迹',
       '享乐主义_该价值观轻微迹象', '享乐主义_该价值观明确表达', '享乐主义_该价值观显著表达', '享乐主义_该价值观深刻贯彻']

# 计算轮数
steps_per_epoch = len(train_df) // PARAMS['BATCH_SIZE']
total_training_steps = steps_per_epoch * PARAMS['EPOCHS']

warmup_steps = total_training_steps // 5

# 划分验证集
train_df, test_df = train_test_split(train_df, test_size=PARAMS["VALIDATION_SET_SIZE"],
                                    random_state=PARAMS["RANDOM_SEED"])

# 读入tokenizer
TOKENIZER = AutoTokenizer.from_pretrained(PARAMS["MODEL_NAME"])

# 注意，train是多分类数据集，因此当val采用valuetalk格式时，val是二分类数据集，两者的准确率会有较大差异
# 另外，由于用于标注的AI本身在valuetalk上准确率不足，因此训练集中存在许多错误
# 导入dataloader
data_module = BertDataModule(
    train_df,
    val_df,
    test_df,
    tokenizer=TOKENIZER,
    params=PARAMS,
    label_columns=LABEL_COLUMNS,
    use_valuetalk=True # 如果使用valuetalk格式数据集作为验证集，请选择true，否则选否
)

# 构建模型
model = DebertaMultiLabelClassifier(params=PARAMS, n_training_steps=total_training_steps, n_warmup_steps=warmup_steps)

# 保存最佳性能
best_checkpoint_callback = ModelCheckpoint(
    dirpath=PARAMS["SAVE_DIR"],
    filename="best-{epoch:02d}-{val_overall_accuracy:.4f}",  # 文件名包含epoch和准确率
    monitor="val_overall_accuracy",  # 监控的指标
    mode="max",  # 最大化指标
    save_top_k=1,  # 只保存最好的1个
    save_last=False,  # 不保存最后一个checkpoint（如果不需要的话）
    verbose=True,
    auto_insert_metric_name=False,  # 不在文件名中自动插入指标名
)
# 保存回调
save_last_checkpoint_callback = ModelCheckpoint(
    dirpath=PARAMS["SAVE_DIR"],
    filename="checkpoints/last_epoch_{epoch}",
    save_last=True,      # 关键参数！
    every_n_epochs=100,
)
# 这个一般不会触发
early_stopping_callback = EarlyStopping(
    monitor=PARAMS["EARLY_STOPPING_METRIC"],
    patience=PARAMS["PATIENCE"],
    mode=PARAMS["EARLY_STOPPING_MODE"]
)




# 设置训练器
trainer = Trainer(
    callbacks=[early_stopping_callback, best_checkpoint_callback, save_last_checkpoint_callback],
    max_epochs=PARAMS["EPOCHS"],
    accelerator="gpu",
    strategy="ddp",
    #strategy="auto",
    devices=7, # 7卡并行
    enable_progress_bar=True,
    check_val_every_n_epoch = 1, # 每轮测试一次，配合保存最佳模型
    accumulate_grad_batches=PARAMS["ACCUMULATE_GRAD_BATCHES"],

)
trainer._should_stop = False
trainer.fit(model, data_module)

################ 以下为参考的训练输出 #######################

"""
(env) root@lyg2239:/home/zxy/for_sync/humanvalue# python code/train.py
/home/zxy/for_sync/humanvalue/code
Seed set to 1
{'USE_COSINE_SCHEDULER': True, 'COSINE_SCHEDULER_TYPE': 'cosine_annealing', 'MIN_LEARNING_RATE': 1e-07, 'WARMUP_STEPS_RATIO': 0.1, 'COSINE_T_MULT': 2, 'COSINE_RESTART_PERIOD': 1, 'NUM_CYCLES': 2, 'TRAIN_PATH': './dataset/train_vt_only_v10_weighted.parquet', 'MODEL_NAME': 'IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese', 'BATCH_SIZE': 16, 'ACCUMULATE_GRAD_BATCHES': 1, 'LEARNING_RATE': 0.0001, 'EPOCHS': 5, 'OPTIMIZER': 'AdamW', 'DEVICE': device(type='cuda'), 'NUM_LABELS': 10, 'NUM_CLASSES_PER_LABEL': 6, 'NUM_TRAIN_WORKERS': 0, 'NUM_VAL_WORKERS': 0, 'MAX_TOKEN_COUNT': 512, 'RANDOM_SEED': 1, 'DROPOUT': 0.2, 'HIDDEN_LAYERS': [(128, ReLU())], 'WEIGHT_DECAY': 0.1, 'CRITERION': [CrossEntropyLoss()], 'PATIENCE': 300, 'VAL_CHECK_INTERVAL': 176, 'MAX_THRESHOLD_METRIC': 'custom', 'EARLY_STOPPING_METRIC': 'val_overall_accuracy', 'EARLY_STOPPING_MODE': 'max', 'SAVE_DIR': '/data1/bert/humanvalue/', 'VALIDATION_SET_SIZE': 100, 'LABEL_NAMES': ['博爱', '友善', '权力', '成就', '传统', '遵从', '安全', '自主', '刺激', '享乐主义']}
Seed set to 1
200
Some weights of the model checkpoint at IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese were not used when initializing DebertaV2Model: ['cls.predictions.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight']
- This IS expected if you are initializing DebertaV2Model from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DebertaV2Model from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
/home/zxy/for_sync/humanvalue/code
/home/zxy/for_sync/humanvalue/code
/home/zxy/for_sync/humanvalue/code
/home/zxy/for_sync/humanvalue/code
/home/zxy/for_sync/humanvalue/code
/home/zxy/for_sync/humanvalue/code
You are using a CUDA device ('NVIDIA A100-SXM4-80GB') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/7
[rank: 5] Seed set to 1
[rank: 1] Seed set to 1
[rank: 6] Seed set to 1
[rank: 2] Seed set to 1
{'USE_COSINE_SCHEDULER': True, 'COSINE_SCHEDULER_TYPE': 'cosine_annealing', 'MIN_LEARNING_RATE': 1e-07, 'WARMUP_STEPS_RATIO': 0.1, 'COSINE_T_MULT': 2, 'COSINE_RESTART_PERIOD': 1, 'NUM_CYCLES': 2, 'TRAIN_PATH': './dataset/train_vt_only_v10_weighted.parquet', 'MODEL_NAME': 'IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese', 'BATCH_SIZE': 16, 'ACCUMULATE_GRAD_BATCHES': 1, 'LEARNING_RATE': 0.0001, 'EPOCHS': 5, 'OPTIMIZER': 'AdamW', 'DEVICE': device(type='cuda'), 'NUM_LABELS': 10, 'NUM_CLASSES_PER_LABEL': 6, 'NUM_TRAIN_WORKERS': 0, 'NUM_VAL_WORKERS': 0, 'MAX_TOKEN_COUNT': 512, 'RANDOM_SEED': 1, 'DROPOUT': 0.2, 'HIDDEN_LAYERS': [(128, ReLU())], 'WEIGHT_DECAY': 0.1, 'CRITERION': [CrossEntropyLoss()], 'PATIENCE': 300, 'VAL_CHECK_INTERVAL': 176, 'MAX_THRESHOLD_METRIC': 'custom', 'EARLY_STOPPING_METRIC': 'val_overall_accuracy', 'EARLY_STOPPING_MODE': 'max', 'SAVE_DIR': '/data1/bert/humanvalue/', 'VALIDATION_SET_SIZE': 100, 'LABEL_NAMES': ['博爱', '友善', '权力', '成就', '传统', '遵从', '安全', '自主', '刺激', '享乐主义']}
[rank: 5] Seed set to 1
[rank: 4] Seed set to 1
[rank: 3] Seed set to 1
{'USE_COSINE_SCHEDULER': True, 'COSINE_SCHEDULER_TYPE': 'cosine_annealing', 'MIN_LEARNING_RATE': 1e-07, 'WARMUP_STEPS_RATIO': 0.1, 'COSINE_T_MULT': 2, 'COSINE_RESTART_PERIOD': 1, 'NUM_CYCLES': 2, 'TRAIN_PATH': './dataset/train_vt_only_v10_weighted.parquet', 'MODEL_NAME': 'IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese', 'BATCH_SIZE': 16, 'ACCUMULATE_GRAD_BATCHES': 1, 'LEARNING_RATE': 0.0001, 'EPOCHS': 5, 'OPTIMIZER': 'AdamW', 'DEVICE': device(type='cuda'), 'NUM_LABELS': 10, 'NUM_CLASSES_PER_LABEL': 6, 'NUM_TRAIN_WORKERS': 0, 'NUM_VAL_WORKERS': 0, 'MAX_TOKEN_COUNT': 512, 'RANDOM_SEED': 1, 'DROPOUT': 0.2, 'HIDDEN_LAYERS': [(128, ReLU())], 'WEIGHT_DECAY': 0.1, 'CRITERION': [CrossEntropyLoss()], 'PATIENCE': 300, 'VAL_CHECK_INTERVAL': 176, 'MAX_THRESHOLD_METRIC': 'custom', 'EARLY_STOPPING_METRIC': 'val_overall_accuracy', 'EARLY_STOPPING_MODE': 'max', 'SAVE_DIR': '/data1/bert/humanvalue/', 'VALIDATION_SET_SIZE': 100, 'LABEL_NAMES': ['博爱', '友善', '权力', '成就', '传统', '遵从', '安全', '自主', '刺激', '享乐主义']}
[rank: 1] Seed set to 1
200
{'USE_COSINE_SCHEDULER': True, 'COSINE_SCHEDULER_TYPE': 'cosine_annealing', 'MIN_LEARNING_RATE': 1e-07, 'WARMUP_STEPS_RATIO': 0.1, 'COSINE_T_MULT': 2, 'COSINE_RESTART_PERIOD': 1, 'NUM_CYCLES': 2, 'TRAIN_PATH': './dataset/train_vt_only_v10_weighted.parquet', 'MODEL_NAME': 'IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese', 'BATCH_SIZE': 16, 'ACCUMULATE_GRAD_BATCHES': 1, 'LEARNING_RATE': 0.0001, 'EPOCHS': 5, 'OPTIMIZER': 'AdamW', 'DEVICE': device(type='cuda'), 'NUM_LABELS': 10, 'NUM_CLASSES_PER_LABEL': 6, 'NUM_TRAIN_WORKERS': 0, 'NUM_VAL_WORKERS': 0, 'MAX_TOKEN_COUNT': 512, 'RANDOM_SEED': 1, 'DROPOUT': 0.2, 'HIDDEN_LAYERS': [(128, ReLU())], 'WEIGHT_DECAY': 0.1, 'CRITERION': [CrossEntropyLoss()], 'PATIENCE': 300, 'VAL_CHECK_INTERVAL': 176, 'MAX_THRESHOLD_METRIC': 'custom', 'EARLY_STOPPING_METRIC': 'val_overall_accuracy', 'EARLY_STOPPING_MODE': 'max', 'SAVE_DIR': '/data1/bert/humanvalue/', 'VALIDATION_SET_SIZE': 100, 'LABEL_NAMES': ['博爱', '友善', '权力', '成就', '传统', '遵从', '安全', '自主', '刺激', '享乐主义']}
[rank: 2] Seed set to 1
{'USE_COSINE_SCHEDULER': True, 'COSINE_SCHEDULER_TYPE': 'cosine_annealing', 'MIN_LEARNING_RATE': 1e-07, 'WARMUP_STEPS_RATIO': 0.1, 'COSINE_T_MULT': 2, 'COSINE_RESTART_PERIOD': 1, 'NUM_CYCLES': 2, 'TRAIN_PATH': './dataset/train_vt_only_v10_weighted.parquet', 'MODEL_NAME': 'IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese', 'BATCH_SIZE': 16, 'ACCUMULATE_GRAD_BATCHES': 1, 'LEARNING_RATE': 0.0001, 'EPOCHS': 5, 'OPTIMIZER': 'AdamW', 'DEVICE': device(type='cuda'), 'NUM_LABELS': 10, 'NUM_CLASSES_PER_LABEL': 6, 'NUM_TRAIN_WORKERS': 0, 'NUM_VAL_WORKERS': 0, 'MAX_TOKEN_COUNT': 512, 'RANDOM_SEED': 1, 'DROPOUT': 0.2, 'HIDDEN_LAYERS': [(128, ReLU())], 'WEIGHT_DECAY': 0.1, 'CRITERION': [CrossEntropyLoss()], 'PATIENCE': 300, 'VAL_CHECK_INTERVAL': 176, 'MAX_THRESHOLD_METRIC': 'custom', 'EARLY_STOPPING_METRIC': 'val_overall_accuracy', 'EARLY_STOPPING_MODE': 'max', 'SAVE_DIR': '/data1/bert/humanvalue/', 'VALIDATION_SET_SIZE': 100, 'LABEL_NAMES': ['博爱', '友善', '权力', '成就', '传统', '遵从', '安全', '自主', '刺激', '享乐主义']}
[rank: 6] Seed set to 1
200
{'USE_COSINE_SCHEDULER': True, 'COSINE_SCHEDULER_TYPE': 'cosine_annealing', 'MIN_LEARNING_RATE': 1e-07, 'WARMUP_STEPS_RATIO': 0.1, 'COSINE_T_MULT': 2, 'COSINE_RESTART_PERIOD': 1, 'NUM_CYCLES': 2, 'TRAIN_PATH': './dataset/train_vt_only_v10_weighted.parquet', 'MODEL_NAME': 'IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese', 'BATCH_SIZE': 16, 'ACCUMULATE_GRAD_BATCHES': 1, 'LEARNING_RATE': 0.0001, 'EPOCHS': 5, 'OPTIMIZER': 'AdamW', 'DEVICE': device(type='cuda'), 'NUM_LABELS': 10, 'NUM_CLASSES_PER_LABEL': 6, 'NUM_TRAIN_WORKERS': 0, 'NUM_VAL_WORKERS': 0, 'MAX_TOKEN_COUNT': 512, 'RANDOM_SEED': 1, 'DROPOUT': 0.2, 'HIDDEN_LAYERS': [(128, ReLU())], 'WEIGHT_DECAY': 0.1, 'CRITERION': [CrossEntropyLoss()], 'PATIENCE': 300, 'VAL_CHECK_INTERVAL': 176, 'MAX_THRESHOLD_METRIC': 'custom', 'EARLY_STOPPING_METRIC': 'val_overall_accuracy', 'EARLY_STOPPING_MODE': 'max', 'SAVE_DIR': '/data1/bert/humanvalue/', 'VALIDATION_SET_SIZE': 100, 'LABEL_NAMES': ['博爱', '友善', '权力', '成就', '传统', '遵从', '安全', '自主', '刺激', '享乐主义']}
[rank: 3] Seed set to 1
{'USE_COSINE_SCHEDULER': True, 'COSINE_SCHEDULER_TYPE': 'cosine_annealing', 'MIN_LEARNING_RATE': 1e-07, 'WARMUP_STEPS_RATIO': 0.1, 'COSINE_T_MULT': 2, 'COSINE_RESTART_PERIOD': 1, 'NUM_CYCLES': 2, 'TRAIN_PATH': './dataset/train_vt_only_v10_weighted.parquet', 'MODEL_NAME': 'IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese', 'BATCH_SIZE': 16, 'ACCUMULATE_GRAD_BATCHES': 1, 'LEARNING_RATE': 0.0001, 'EPOCHS': 5, 'OPTIMIZER': 'AdamW', 'DEVICE': device(type='cuda'), 'NUM_LABELS': 10, 'NUM_CLASSES_PER_LABEL': 6, 'NUM_TRAIN_WORKERS': 0, 'NUM_VAL_WORKERS': 0, 'MAX_TOKEN_COUNT': 512, 'RANDOM_SEED': 1, 'DROPOUT': 0.2, 'HIDDEN_LAYERS': [(128, ReLU())], 'WEIGHT_DECAY': 0.1, 'CRITERION': [CrossEntropyLoss()], 'PATIENCE': 300, 'VAL_CHECK_INTERVAL': 176, 'MAX_THRESHOLD_METRIC': 'custom', 'EARLY_STOPPING_METRIC': 'val_overall_accuracy', 'EARLY_STOPPING_MODE': 'max', 'SAVE_DIR': '/data1/bert/humanvalue/', 'VALIDATION_SET_SIZE': 100, 'LABEL_NAMES': ['博爱', '友善', '权力', '成就', '传统', '遵从', '安全', '自主', '刺激', '享乐主义']}
[rank: 4] Seed set to 1
200
200
200
200
Some weights of the model checkpoint at IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese were not used when initializing DebertaV2Model: ['cls.predictions.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight']
- This IS expected if you are initializing DebertaV2Model from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DebertaV2Model from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of the model checkpoint at IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese were not used when initializing DebertaV2Model: ['cls.predictions.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight']
- This IS expected if you are initializing DebertaV2Model from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DebertaV2Model from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Initializing distributed: GLOBAL_RANK: 5, MEMBER: 6/7
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/7
Some weights of the model checkpoint at IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese were not used when initializing DebertaV2Model: ['cls.predictions.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight']
- This IS expected if you are initializing DebertaV2Model from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DebertaV2Model from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/7
Some weights of the model checkpoint at IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese were not used when initializing DebertaV2Model: ['cls.predictions.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight']
- This IS expected if you are initializing DebertaV2Model from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DebertaV2Model from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of the model checkpoint at IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese were not used when initializing DebertaV2Model: ['cls.predictions.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight']
- This IS expected if you are initializing DebertaV2Model from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DebertaV2Model from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of the model checkpoint at IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese were not used when initializing DebertaV2Model: ['cls.predictions.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight']
- This IS expected if you are initializing DebertaV2Model from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DebertaV2Model from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Initializing distributed: GLOBAL_RANK: 6, MEMBER: 7/7
Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/7
Initializing distributed: GLOBAL_RANK: 4, MEMBER: 5/7
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 7 processes
----------------------------------------------------------------------------------------------------

启用加权损失
启用加权损失
启用加权损失
使用普通采样
使用普通采样
使用普通采样
启用加权损失
启用加权损失
使用普通采样
使用普通采样
启用加权损失
使用普通采样
启用加权损失
使用普通采样
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [1,2,3,4,5,6,7]
LOCAL_RANK: 6 - CUDA_VISIBLE_DEVICES: [1,2,3,4,5,6,7]
LOCAL_RANK: 4 - CUDA_VISIBLE_DEVICES: [1,2,3,4,5,6,7]
LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [1,2,3,4,5,6,7]
LOCAL_RANK: 5 - CUDA_VISIBLE_DEVICES: [1,2,3,4,5,6,7]
LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [1,2,3,4,5,6,7]
LOCAL_RANK: 2 - CUDA_VISIBLE_DEVICES: [1,2,3,4,5,6,7]
使用余弦学习率
使用余弦学习率
使用余弦学习率
使用余弦学习率
使用余弦学习率
使用余弦学习率
使用余弦学习率
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name          ┃ Type                     ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ bert          │ DebertaV2Model           │  319 M │ train │     0 │
│ 1 │ hidden_layers │ ModuleList               │  131 K │ train │     0 │
│ 2 │ classifier    │ Linear                   │  7.7 K │ train │     0 │
│ 3 │ criterion     │ WeightedCrossEntropyLoss │      0 │ train │     0 │
│ 4 │ test_auroc    │ MultilabelAUROC          │      0 │ train │     0 │
│ 5 │ val_accuracy  │ MultilabelAccuracy       │      0 │ train │     0 │
└───┴───────────────┴──────────────────────────┴────────┴───────┴───────┘
Trainable params: 243 M                                                                                                                                                                                                                 
Non-trainable params: 75.6 M                                                                                                                                                                                                            
Total params: 319 M                                                                                                                                                                                                                     
Total estimated model params size (MB): 1.3 K                                                                                                                                                                                           
Modules in train mode: 477                                                                                                                                                                                                              
Modules in eval mode: 0                                                                                                                                                                                                                 
Total FLOPs: 0                                                                                                                                                                                                                          
Epoch 0/4  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 88/88 0:06:02 • 0:00:00 0.27it/s v_num: 599.000 train_loss: 22.037Epoch 0, global step 88: 'val_overall_accuracy' reached 0.72709 (best 0.72709), saving model to '/data1/bert/humanvalue/best-00-0.7271.ckpt' as top 1
Epoch 1/4  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 88/88 0:06:02 • 0:00:00 0.27it/s v_num: 599.000 train_loss: 14.262 val_overall_accuracy: 0.727 train_epoch_loss: 30.714Epoch 1, global step 176: 'val_overall_accuracy' reached 0.79606 (best 0.79606), saving model to '/data1/bert/humanvalue/best-01-0.7961.ckpt' as top 1
Epoch 2/4  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 88/88 0:06:02 • 0:00:00 0.27it/s v_num: 599.000 train_loss: 24.322 val_overall_accuracy: 0.796 train_epoch_loss: 21.943Epoch 2, global step 264: 'val_overall_accuracy' reached 0.85665 (best 0.85665), saving model to '/data1/bert/humanvalue/best-02-0.8567.ckpt' as top 1
Epoch 3/4  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 88/88 0:06:02 • 0:00:00 0.27it/s v_num: 599.000 train_loss: 11.242 val_overall_accuracy: 0.857 train_epoch_loss: 17.547Epoch 3, global step 352: 'val_overall_accuracy' reached 0.86158 (best 0.86158), saving model to '/data1/bert/humanvalue/best-03-0.8616.ckpt' as top 1
Epoch 4/4  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 88/88 0:06:02 • 0:00:00 0.27it/s v_num: 599.000 train_loss: 12.904 val_overall_accuracy: 0.862 train_epoch_loss: 14.194Epoch 4, global step 440: 'val_overall_accuracy' reached 0.86453 (best 0.86453), saving model to '/data1/bert/humanvalue/best-04-0.8645.ckpt' as top 1
`Trainer.fit` stopped: `max_epochs=5` reached.
Epoch 4/4  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 88/88 0:06:02 • 0:00:00 0.27it/s v_num: 599.000 train_loss: 12.904 val_overall_accuracy: 0.865 train_epoch_loss: 14.194
"""
