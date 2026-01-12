from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import torch


class BertDataModule(LightningDataModule):

    def __init__(self, train_df, val_df, test_df, tokenizer, params, label_columns, use_valuetalk=False):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.params = params
        self.tokenizer = tokenizer
        self.label_columns = label_columns
        self.use_valuetalk = use_valuetalk

        # 检查训练数据是否包含权重列
        self.has_weights = 'data_weight' in train_df.columns

    def setup(self, stage=None):
        self.train_dataset = BertDataset(
            data=self.train_df,
            tokenizer=self.tokenizer,
            max_token_count=self.params["MAX_TOKEN_COUNT"],
            label_columns=self.label_columns,
            include_weights=self.has_weights  # 传递权重信息
        )

        self.val_dataset = BertDataset(
            data=self.val_df,
            tokenizer=self.tokenizer,
            max_token_count=self.params["MAX_TOKEN_COUNT"],
            label_columns=self.label_columns,
            use_valuetalk=self.use_valuetalk,  # 是否使用value talk数据集，这个数据集的格式不太一样
        )

        self.test_dataset = BertDataset(
            data=self.test_df,
            tokenizer=self.tokenizer,
            max_token_count=self.params["MAX_TOKEN_COUNT"],
            label_columns=self.label_columns,
        )

        # 创建加权采样器（如果训练数据有权重）
        if self.has_weights and False:
            weights = self.train_df['data_weight'].values
            # 归一化权重用于采样器
            weights = weights / weights.sum() #权重总和归一化
            self.train_df['data_weight'] = weights * len(weights) # 希望所有样本的权重综合还是1*len，这样所有样本的损失求和理论上一致
            self.weighted_sampler = WeightedRandomSampler( # 加权采样器
                weights=weights,
                num_samples=len(weights),
                replacement=True
            )
            print(f"创建加权采样器: 权重范围 {weights.min():.4f} - {weights.max():.4f}")
        else:
            self.weighted_sampler = None
            print("使用普通采样")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.params["BATCH_SIZE"],
            shuffle=True,
            num_workers=self.params["NUM_TRAIN_WORKERS"]
        )

    def train_dataloader_sample(self):
        # 启用加权采样器的版本
        print("启用加权采样器")
        if self.has_weights and self.weighted_sampler:
            # 使用加权采样器（不要同时shuffle=True）
            return DataLoader(
                self.train_dataset,
                batch_size=self.params["BATCH_SIZE"],
                sampler=self.weighted_sampler,  # 使用采样器而不是shuffle
                num_workers=self.params["NUM_TRAIN_WORKERS"],
                persistent_workers=True if self.params["NUM_TRAIN_WORKERS"] > 0 else False
            )
        else:
            # 回退到普通采样
            return DataLoader(
                self.train_dataset,
                batch_size=self.params["BATCH_SIZE"],
                shuffle=True,
                num_workers=self.params["NUM_TRAIN_WORKERS"],
                persistent_workers=True if self.params["NUM_TRAIN_WORKERS"] > 0 else False
            )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.params["BATCH_SIZE"],
            num_workers=self.params["NUM_VAL_WORKERS"]
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.params["BATCH_SIZE"],
            num_workers=self.params["NUM_VAL_WORKERS"]
        )


class BertDataset(Dataset):

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer,
            max_token_count,
            label_columns=None,
            use_valuetalk=False,
            include_weights=False  # 新增参数：是否包含权重
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_count
        if label_columns:
            self.label_columns = label_columns
        else:
            self.label_columns = None
        self.use_valuetalk = use_valuetalk

        self.include_weights = include_weights and 'data_weight' in data.columns
        if self.include_weights:
            print("启用加权损失")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        text = data_row.text
        # 不同数据集的格式不一样
        if self.use_valuetalk:
            valuetalk_col = ["博爱","友善","权力","成就","传统","遵从","安全","自主","刺激","享乐主义"]


            labels = data_row[valuetalk_col].values.astype(float)
        if self.label_columns and not self.use_valuetalk:
            labels = data_row[self.label_columns].values.astype(float)

        # 编码处理
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        if self.label_columns:
            result = dict(
                text=text,
                input_ids=encoding["input_ids"].flatten(),
                attention_mask=encoding["attention_mask"].flatten(),
                labels=torch.FloatTensor(labels)
            )
        else:
            result = dict(
                text=text,
                input_ids=encoding["input_ids"].flatten(),
                attention_mask=encoding["attention_mask"].flatten()
            )
        # 增加权重
        if self.include_weights:
            weight = data_row['data_weight']
            result["weight"] = torch.tensor(weight, dtype=torch.float)
            result["index"] = torch.tensor(index, dtype=torch.long)  # 用于调试

        return result

