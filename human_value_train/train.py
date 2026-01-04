import os
import sys

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
sys.path.append(current_dir)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch.nn as nn
from data.DataModule import BertDataModule
from model.IntergrateBERT import DebertaMultiLabelClassifier
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 2025

seed_everything(RANDOM_SEED)

PARAMS = {
    # Language Model and Hyperparameters
    "MODEL_NAME": 'IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese',
    #"BATCH_SIZE": 16,
    "BATCH_SIZE": 16,
    "ACCUMULATE_GRAD_BATCHES": 1,
    "LEARNING_RATE": 5e-5,
    "EPOCHS": 7,
    #"EPOCHS": 50,
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
    "HIDDEN_LAYERS": [(256, nn.ReLU())], # 没有武汉数据的则采用128即可
    "WEIGHT_DECAY": 0.1,
    "CRITERION": [nn.CrossEntropyLoss()],

    # Early Stopping Params
    "PATIENCE": 300,
    "VAL_CHECK_INTERVAL": 62,

    # The metric we optimize for. Alternative "custom_f1/Val" and "max"
    "MAX_THRESHOLD_METRIC": "custom",  # The f1-score that should maximized (custom = formula for the task evaluation)
    "EARLY_STOPPING_METRIC": "val_overall_accuracy",
    "EARLY_STOPPING_MODE": "max",

    # DATA
    "VALIDATION_SET_SIZE": 100,
    #"TRAIN_PATH": "./dataset/training_data_with_weights_clean_kfold.parquet",  #
    "TRAIN_PATH": "./dataset/wuhan_train_weighted.parquet",
    #"TRAIN_PATH": "./dataset/training_data_fix.parquet",
    "LABEL_NAMES":[
        "博爱", "友善", "权力", "成就", "传统",
        "遵从", "安全", "自主", "刺激", "享乐主义"
    ],
}
print(PARAMS)
# 读入训练集
train_df = pd.read_parquet(PARAMS["TRAIN_PATH"])

#LABEL_COLUMNS = train_df.columns.tolist()[1:]
# 确定test数据的label维度
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
#warmup_steps = 10 // 5
# 划分验证集
train_df, val_df = train_test_split(train_df, test_size=PARAMS["VALIDATION_SET_SIZE"],
                                    random_state=PARAMS["RANDOM_SEED"])

# 读入tokenizer
# TOKENIZER = AutoTokenizer.from_pretrained(PARAMS["MODEL_PATH"])
TOKENIZER = AutoTokenizer.from_pretrained(PARAMS["MODEL_NAME"])
# 导入dataloader
data_module = BertDataModule(
    train_df,
    val_df,
    val_df,
    tokenizer=TOKENIZER,
    params=PARAMS,
    label_columns=LABEL_COLUMNS,
)

# 构建模型
model = DebertaMultiLabelClassifier(params=PARAMS, n_training_steps=total_training_steps, n_warmup_steps=warmup_steps)

# 保存回调
save_last_checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",
    filename="last_epoch_{epoch}",
    save_last=True,      # 关键参数！
    every_n_epochs=100,
)
early_stopping_callback = EarlyStopping(
    monitor=PARAMS["EARLY_STOPPING_METRIC"],
    patience=PARAMS["PATIENCE"],
    mode=PARAMS["EARLY_STOPPING_MODE"]
)




# 设置训练器
trainer = Trainer(
    callbacks=[early_stopping_callback, save_last_checkpoint_callback],
    max_epochs=PARAMS["EPOCHS"],
    # fast_dev_run=True,
    accelerator="gpu",
    strategy="ddp", #取消分布式
    #strategy="auto",
    devices=1, # 7卡并行
    enable_progress_bar=True,
    val_check_interval=PARAMS["VAL_CHECK_INTERVAL"],
    accumulate_grad_batches=PARAMS["ACCUMULATE_GRAD_BATCHES"],

)
trainer._should_stop = False
trainer.fit(model, data_module)
