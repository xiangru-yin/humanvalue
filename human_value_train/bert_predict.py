# predict.py
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version built with: {torch.version.cuda}")  # 如果返回None，说明是CPU版本
import random
import json
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from lightning import Trainer
import warnings
hanzi_value_dict = {
    "该价值观毫无体现": 0.0,
    "该价值观极微弱痕迹": 0.1,
    "该价值观轻微迹象": 0.3,
    "该价值观明确表达": 0.5,
    "该价值观显著表达": 0.8,
    "该价值观深刻贯彻": 1.0,

    0:0.0,
    1:0.1,
    2:0.3,
    3:0.5,
    4:0.8,
    5:1.0
}
# 设置环境变量必须在导入transformers之前

from model.IntergrateBERT_predict import DebertaMultiLabelClassifier
from data.DataModule import BertDataset
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')

# 全局变量
_model = None
_tokenizer = None
_trainer = None
_params = None
_label_columns = None
_device = None
_batch_size = None
_max_token_count = None
_num_workers = None


def init(checkpoint_path="humanvalue_api/checkpoints/last_128_10epoch_weightdecay01_801.ckpt",
         batch_size=32,
         max_token_count=512,
         num_workers=0,
         device=None,
         model_name='IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese',
         use_local_model=True,
         #local_model_path='../human_value_predict/humanvalue_api/deberta_tokenizer',
         local_model_path='IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese'):
    """
    初始化价值观预测器
    """
    global _model, _tokenizer, _trainer, _params, _label_columns, _device, _batch_size, _max_token_count, _num_workers

    print("="*50)
    print("初始化价值观预测器...")
    print("="*50)

    # 设置随机种子
    random_seed = 2025
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # 设备
    if device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        _device = device
    print(f"使用设备: {_device}")

    # 参数
    _batch_size = batch_size
    _max_token_count = max_token_count
    _num_workers = num_workers

    _params = {
        "MODEL_NAME": model_name,
        "BATCH_SIZE": batch_size,
        "MAX_TOKEN_COUNT": max_token_count,
        "RANDOM_SEED": random_seed,
        "NUM_LABELS": 10,
        "NUM_CLASSES_PER_LABEL": 6,
        "DEVICE": _device,
        "DROPOUT": 0.1,
        "HIDDEN_LAYERS": [(128, nn.ReLU())],
        "LEARNING_RATE": 2e-5,
        "EPOCHS": 0,
        "LABEL_NAMES": ['博爱', '友善', '权力', '成就', '传统', '遵从', '安全', '自主', '刺激', '享乐主义']
    }

    # tokenizer
    print(f"加载tokenizer: {model_name}")
    if use_local_model and local_model_path:
        _tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            local_files_only=True,
            trust_remote_code=True
        )
    else:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                mirror='https://hf-mirror.com',
                trust_remote_code=True,
                use_fast=True
            )
        except Exception:
            from transformers import BertTokenizerFast
            _tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    print("Tokenizer加载成功！")

    # 模型
    print(f"加载模型权重: {checkpoint_path}")
    _model = DebertaMultiLabelClassifier.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        params=_params
    )

    _model.eval()
    _model.to(_device)
    print("模型加载成功！")

    # Trainer 仅用于预测
    _trainer = Trainer(accelerator="auto", devices=1, enable_checkpointing=False)

    print("="*50)
    print("初始化完成！")
    print("="*50)

def get_value_ensemble_dir_bert(dir_path, skip_used = True):
    def get_files_in_folder(folder_path):
        # 获取所有的csv数据样本，并返回csv_list
        file_list = []
        for root, dirs, files in os.walk(folder_path):
            for file_path in files:
                if "csv" in file_path:
                    file_list.append(os.path.join(root, file_path))
        return file_list
    csv_list = get_files_in_folder(dir_path)
    total_len, filter_len = 1, 1
    for data_path in csv_list:
        if 'ensemble' in data_path:
            continue
        if "split" not in data_path:
            continue
        try:
            forward_file(event_data_csv_path=data_path, event_image_dir="", skip_used=skip_used)
        except Exception as e:
            print(f"汇总失败，文件：{data_path}，原因：{e}")
    return dir_path


def forward_file(event_data_csv_path="",
            event_image_dir="", skip_used = False):
    """
    进行预测
    """
    global _model, _tokenizer, _trainer, _params, _label_columns, _device, _batch_size, _max_token_count, _num_workers

    if _model is None or _tokenizer is None or _trainer is None:
        raise RuntimeError("请先调用init()函数初始化模型")

    print("\n" + "="*50)
    print("="*50)

    if not event_data_csv_path or not os.path.exists(event_data_csv_path):
        raise ValueError(f"输入CSV文件不存在: {event_data_csv_path}")

    post_result_csv_path = event_data_csv_path.replace('.csv', '_ensemble.csv')
    if os.path.exists(post_result_csv_path) and skip_used:
        print("跳过结果")
        pass
        return None, post_result_csv_path


    # 数据
    if event_data_csv_path.endswith('.csv'):
        predict_df = pd.read_csv(event_data_csv_path)
    elif event_data_csv_path.endswith('.parquet'):
        predict_df = pd.read_parquet(event_data_csv_path)
    else:
        predict_df = pd.read_csv(event_data_csv_path)

    if 'text' not in predict_df.columns:
        raise ValueError("输入数据必须包含'text'列")
    if '大模型汇总价值观' in predict_df.columns:
        predict_df['大模型汇总价值观'] = ""
    predict_df = predict_df[predict_df['text']!=""]
    predict_df = predict_df.reset_index(drop=True)
    # 打包成DalaLoader
    predict_dataset = BertDataset(
        predict_df,
        tokenizer=_tokenizer,
        max_token_count=_max_token_count,
        label_columns=_label_columns
    )

    predict_dataloader = DataLoader(
        predict_dataset,
        batch_size=_batch_size,
        shuffle=False,
        num_workers=_num_workers
    )

    # 预测
    predictions = _trainer.predict(_model, dataloaders=predict_dataloader)
    if not predictions:
        raise RuntimeError("预测失败，未获得预测结果")

    # 重新分发帖子级结果
    post_results = []
    all_predictions = []

    for idx, pred in enumerate(predictions):
        probabilities = pred["binary_probabilities"]
        predicted_classes = pred["predictions"]
        multi_classes = pred["multi_predictions"]
        for batch_idx in range(predicted_classes.shape[0]):
            sample_idx = idx * _batch_size + batch_idx
            if sample_idx >= len(predict_df):
                break

            sample_result = {
                'index': sample_idx,
                'text': predict_df.iloc[sample_idx]['text']
            }
            ensemble_value_dict = {}
            for label_idx in range(_params["NUM_LABELS"]):
                # 解析结果
                label_name = _params["LABEL_NAMES"][label_idx]
                pred_class = predicted_classes[batch_idx, label_idx].item()
                prob = probabilities[batch_idx, label_idx].item()
                multi_label = multi_classes[batch_idx, label_idx].item()
                multi_label = hanzi_value_dict[multi_label]
                sample_result[f'{label_name}_predicted'] = pred_class
                sample_result[f'{label_name}_prob'] = prob
                sample_result[f'{label_name}_multi_label'] = multi_label

                # 保存结果
                ensemble_value_dict[f'{label_name}'] = prob # 更换为二分类概率，适配valuetalk

            # 依然保存为大模型汇总价值观数据列
            sample_result[f'大模型汇总价值观'] = str(ensemble_value_dict)
            predict_df.loc[sample_idx, '大模型汇总价值观'] = str(ensemble_value_dict)
            post_results.append(sample_result)
            label_predictions = [predicted_classes[batch_idx, i].item() for i in range(_params["NUM_LABELS"])]
            all_predictions.append(label_predictions)


    predict_df = predict_df[predict_df['大模型汇总价值观']!=""]
    predict_df = predict_df.reset_index(drop=True)
    predict_df.to_csv(post_result_csv_path, index=False, encoding='utf_8_sig')
    print(f"帖子级结果已保存至: {post_result_csv_path}")

    # 事件级结果

    print("\n" + "="*50)
    print("预测完成！")
    print(f"- 帖子级结果: {post_result_csv_path}")
    print("="*50)

    return None, post_result_csv_path


if __name__ == "__main__":
    init(
        checkpoint_path="/data1/bert/humanvalue/best-04-0.8645.ckpt", # 你刚训练好的模型，可以用最后一轮，或者最佳一轮
        batch_size=16,
        max_token_count=512
    )
    forward_file("/home/zxy/for_sync/humanvalue/dataset/valuetalk_test_only_v10.csv", "")
