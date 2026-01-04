import os
from pathlib import Path

import torch
import torch.nn as nn
from lightning import LightningModule
from transformers import AutoModel

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class DebertaMultiLabelClassifier(LightningModule):
    def __init__(self, params):
        super().__init__()
        # 参数
        self.params = params
        self.max_length = params["MAX_TOKEN_COUNT"]
        self.batch_size = params["BATCH_SIZE"]

        # 加载预训练模型
        print(params["MODEL_NAME"])
        self.bert = AutoModel.from_pretrained(params["MODEL_NAME"], return_dict=True)
        # 保存模型到本地目录
        #save_path = "../human_value_predict/saved_models"  # 替换为你想要的保存路径
        #self.bert.save_pretrained(save_path)
        last_output = self.bert.config.hidden_size

        # 构建隐藏层
        self.hidden_layers = nn.ModuleList()
        if params["HIDDEN_LAYERS"]:
            for (h_layer_size, activation) in params["HIDDEN_LAYERS"]:
                if params["DROPOUT"]:
                    self.hidden_layers.append(nn.Dropout(params['DROPOUT']))
                if activation is not None:
                    self.hidden_layers.append(nn.Linear(last_output, h_layer_size))
                self.hidden_layers.append(activation)
                last_output = h_layer_size

        if params["DROPOUT"] and (params["HIDDEN_LAYERS"] is None):
            self.hidden_layers.append(nn.Dropout(params['DROPOUT']))

        # 分类器
        self.classifier = nn.Linear(
            last_output,
            params["NUM_LABELS"] * params["NUM_CLASSES_PER_LABEL"]
        )

        self.n_labels = params["NUM_LABELS"]
        self.n_classes_per_label = params["NUM_CLASSES_PER_LABEL"]
        self.label_names = params["LABEL_NAMES"]

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = output.last_hidden_state

        # 提取 [CLS]
        cls_embedding = last_hidden_state[:, 0, :]

        x = cls_embedding
        for layer in self.hidden_layers:
            x = layer(x)

        logits = self.classifier(x).view(-1, self.n_labels, self.n_classes_per_label)
        probs = torch.softmax(logits, dim=-1)

        return probs

    def predict_step(self, batch, batch_idx=None):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # 得到完整的多分类概率
        probs_full = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        # probs_full: (batch_size, n_labels, n_classes_per_label)

        # === 多标签二分类重构 ===
        # 前一半类别合并为负类 (0)，后一半类别合并为正类 (1)
        prob_class0 = probs_full[..., :3].sum(dim=-1)  # (bs, n_labels)
        prob_class1 = probs_full[..., 3:].sum(dim=-1)  # (bs, n_labels)

        # 正类概率
        binary_probs = prob_class1  # (bs, n_labels)

        # 根据阈值 0.5 进行预测
        predictions = (binary_probs > 0.5).long()  # (bs, n_labels)

        # 多分类结果
        multi_predictions = probs_full.argmax(axis=2)
        return {
            "binary_probabilities": binary_probs,  # 正类概率
            "predictions": predictions,            # 二分类预测 (0/1)
            "multi_predictions": multi_predictions
        }
    def predict_step_cyx(self, batch, batch_idx=None):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # 得到完整的多分类概率
        probs_full = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        # probs_full: (batch_size, n_labels, n_classes_per_label)

        # === 多标签二分类重构 ===
        # 前一半类别合并为负类 (0)，后一半类别合并为正类 (1)
        prob_class0 = probs_full[..., :self.n_classes_per_label // 2].sum(dim=-1)  # (bs, n_labels)
        prob_class1 = probs_full[..., self.n_classes_per_label // 2:].sum(dim=-1)  # (bs, n_labels)

        # 正类概率
        binary_probs = prob_class1  # (bs, n_labels)

        # 根据阈值 0.5 进行预测
        predictions = (binary_probs > 0.5).long()  # (bs, n_labels)

        return {
            "binary_probabilities": binary_probs,  # 正类概率
            "predictions": predictions             # 二分类预测 (0/1)
        }

    # 辅助方法：获取标签名
    def _get_label_name(self, index):
        if self.label_names and index < len(self.label_names):
            return self.label_names[index]
        return f"label_{index}"
