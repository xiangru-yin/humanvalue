import os

import numpy as np
from lightning import LightningModule
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
from torch.optim import AdamW
from torchmetrics import ConfusionMatrix, F1Score, Accuracy, AUROC

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
from transformers import DebertaForSequenceClassification, get_linear_schedule_with_warmup, AutoModel

import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, x, target, weight=None):
        """
        x: 预测logits, 形状 [batch_size, num_classes]
        target: one-hot 编码的目标标签, 形状 [batch_size, num_classes]
        weight: 样本权重, 形状 [batch_size]
        """
        # 将 one-hot 转换为类别索引
        target_indices = torch.argmax(target, dim=1)

        # 使用标准的标签平滑交叉熵
        log_probs = F.log_softmax(x, dim=-1)

        # 计算负对数似然损失
        nll_loss = -log_probs.gather(dim=-1, index=target_indices.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)

        # 计算平滑损失
        smooth_loss = -log_probs.mean(dim=-1)

        # 组合损失
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        # 应用样本权重
        if weight is not None:
            # 确保权重形状与损失匹配 [batch_size]
            if weight.dim() == 1 and weight.shape[0] == loss.shape[0]:
                loss = loss * weight
            else:
                print(f"权重形状不匹配: loss {loss.shape}, weight {weight.shape}")

        # 根据reduction参数返回
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x, target, weight=None):
        """
        x: 预测logits, 形状 [batch_size, num_classes]
        target: one-hot 编码的目标标签, 形状 [batch_size, num_classes]
        weight: 样本权重, 形状 [batch_size]
        """
        # 将 one-hot 转换为类别索引
        target_indices = torch.argmax(target, dim=1)

        # 计算标准交叉熵损失
        loss = F.cross_entropy(x, target_indices, reduction='none')

        # 应用样本权重
        if weight is not None:
            # 确保权重形状与损失匹配 [batch_size]
            if weight.dim() == 1 and weight.shape[0] == loss.shape[0]:
                loss = loss * weight
            else:
                print(f"权重形状不匹配: loss {loss.shape}, weight {weight.shape}")

        # 根据reduction参数返回
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
class DebertaMultiLabelClassifier(LightningModule):
    def __init__(self, params, n_training_steps=None, n_warmup_steps=None, label_smoothing=0):
        super().__init__()
        # 从params字典提取参数
        self.params = params
        self.learning_rate = params["LEARNING_RATE"]
        self.max_length = params["MAX_TOKEN_COUNT"]
        self.batch_size = params["BATCH_SIZE"]
        # 加载预训练DeBERTa模型
        self.bert = AutoModel.from_pretrained(params["MODEL_NAME"], return_dict=True)
        if True:
            self.bert.train()  # 设置为训练模式

            # 确保所有参数可训练
            for param in self.bert.parameters():
                param.requires_grad = True  # 解冻所有BERT参数
        # 冻结BERT的前几层，只微调后面几层（减少过拟合噪声）
        if True:
            freeze_layers = 6
            for name, param in self.bert.named_parameters():
                if any(f'layer.{i}.' in name for i in range(freeze_layers)):
                    param.requires_grad = False
        last_output = self.bert.config.hidden_size
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

        self.classifier = nn.Linear(last_output, params["NUM_LABELS"] * params["NUM_CLASSES_PER_LABEL"])

        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

        self.label_smoothing = label_smoothing
        if label_smoothing <= 0:
            self.criterion = WeightedCrossEntropyLoss()
        else:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.n_labels = params["NUM_LABELS"]
        self.n_classes_per_label = params["NUM_CLASSES_PER_LABEL"]
        self.label_names = params["LABEL_NAMES"]

        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_probs = []
        self.test_targets = []

        # 初始化AUROC指标（多标签）
        self.test_auroc = AUROC(task='multilabel', num_labels=params["NUM_LABELS"])

        # 使用torchmetrics，自动处理DDP同步
        self.val_accuracy = Accuracy(
            task='multilabel',
            num_labels=params["NUM_LABELS"],
            average='micro'  # 微平均，与您的计算一致
        )


    def forward(self, input_ids, attention_mask, labels=None, weight=None):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = output.last_hidden_state

        # 提取[CLS]标记
        cls_embedding = last_hidden_state[:, 0, :]

        x = cls_embedding
        for layer in self.hidden_layers:
            x = layer(x)

        logits = self.classifier(x).view(-1, self.n_labels, self.n_classes_per_label)
        probs = torch.softmax(logits, dim=-1)

        loss = 0
        for i in range(self.n_labels):
            pred = logits[:, i, :].view(-1, self.n_classes_per_label)
            true = labels[:, i, :].view(-1, self.n_classes_per_label)
            if weight is not None:
                loss += self.criterion(pred, true, weight=weight)
            else:
                loss += self.criterion(pred, true)

        return loss, probs

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].reshape(-1, self.n_labels, self.n_classes_per_label)
        # 前向传播
        if "weight" in batch:
            loss, outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels, weight=batch["weight"])
        else:
            loss, outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        self.log('train_loss', loss, prog_bar=True)
        self.training_step_outputs.append({"loss": loss, "predictions": outputs, "labels": labels})
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def on_train_epoch_end(self):
        # 计算整个epoch的平均损失
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        self.log('train_epoch_loss', avg_loss, prog_bar=True, logger=True)

        # 收集所有预测和标签
        all_preds = []
        all_labels = []

        for output in self.training_step_outputs:
            # 获取预测类别索引并移动到CPU
            preds = torch.argmax(output['predictions'], dim=-1).cpu()
            labels = torch.argmax(output['labels'], dim=-1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

        # 拼接所有批次的预测和标签
        if all_preds:  # 确保列表不为空
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # 计算整体准确率
            overall_accuracy = Accuracy(task='multiclass', num_classes=self.n_classes_per_label, average='macro')(
                all_preds.view(-1), all_labels.view(-1)
            )
            self.log('train_overall_accuracy', overall_accuracy, prog_bar=True, logger=True)

            # 计算每个标签的准确率和F1分数
            for label_idx in range(self.n_labels):
                # 单个标签的准确率
                label_acc = Accuracy(task='multiclass', num_classes=self.n_classes_per_label)(
                    all_preds[:, label_idx], all_labels[:, label_idx]
                )
                self.log(f'train_label_{label_idx}_accuracy', label_acc, logger=True)

                # 单个标签的F1分数（宏平均）
                label_f1 = F1Score(task='multiclass', num_classes=self.n_classes_per_label, average='macro')(
                    all_preds[:, label_idx], all_labels[:, label_idx]
                )
                self.log(f'train_label_{label_idx}_f1_macro', label_f1, logger=True)
        else:
            print("Warning: No training outputs collected for epoch end")

        # 清空训练步骤输出
        self.training_step_outputs.clear()

    def validation_step_save(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].reshape(-1, self.n_labels, self.n_classes_per_label)
        # 前向传播
        loss, outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.validation_step_outputs.append({"val_loss": loss, "predictions": outputs, "labels": labels})
        return {"val_loss": loss, "predictions": outputs, "labels": labels}


    def on_validation_epoch_end_saved(self):
        # 计算整个epoch的平均验证损失
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        self.log('val_epoch_loss', avg_loss, prog_bar=True, logger=True)

        # 收集所有预测和标签
        all_preds = []
        all_labels = []

        for output in self.validation_step_outputs:
            # 获取预测类别索引并移动到CPU
            preds = torch.argmax(output['predictions'], dim=-1).cpu()
            labels = torch.argmax(output['labels'], dim=-1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

        # 拼接所有批次的预测和标签
        if all_preds:  # 确保列表不为空
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # 计算整体准确率
            overall_accuracy = Accuracy(task='multiclass', num_classes=self.n_classes_per_label, average='macro')(
                all_preds.view(-1), all_labels.view(-1)
            )
            self.log('val_overall_accuracy', overall_accuracy, prog_bar=True, logger=True)

            # 计算每个标签的准确率和F1分数
            for label_idx in range(self.n_labels):
                # 单个标签的准确率
                label_acc = Accuracy(task='multiclass', num_classes=self.n_classes_per_label)(
                    all_preds[:, label_idx], all_labels[:, label_idx]
                )
                self.log(f'val_label_{label_idx}_accuracy', label_acc, logger=True)

                # 单个标签的F1分数（宏平均）
                label_f1 = F1Score(task='multiclass', num_classes=self.n_classes_per_label, average='macro')(
                    all_preds[:, label_idx], all_labels[:, label_idx]
                )
                self.log(f'val_label_{label_idx}_f1_macro', label_f1, logger=True)

                # 计算并记录混淆矩阵（仅在最后一个标签记录以减少日志量）
                if label_idx == self.n_labels - 1:
                    confmat = ConfusionMatrix(task='multiclass', num_classes=self.n_classes_per_label)(
                        all_preds[:, label_idx], all_labels[:, label_idx]
                    )
                    for i in range(self.n_classes_per_label):
                        for j in range(self.n_classes_per_label):
                            self.log(f'val_label_{label_idx}_confmat_{i}_{j}', confmat[i, j], logger=True)

            # 生成分类报告
            # if self.current_epoch % 5 == 0:  # 每5个epoch记录一次详细报告
            #     # 转换为numpy数组
            #     preds_np = all_preds.numpy()
            #     labels_np = all_labels.numpy()
            #
            #     # 生成分类报告
            #     report = classification_report(
            #         labels_np.reshape(-1),
            #         preds_np.reshape(-1),
            #         target_names=[f'label_{i}_class_{j}' for i in range(self.n_labels) for j in range(self.n_classes_per_label)],
            #         output_dict=True
            #     )
            #
            #     # 记录主要指标
            #     self.log('val_macro_avg_precision', report['macro avg']['precision'], logger=True)
            #     self.log('val_macro_avg_recall', report['macro avg']['recall'], logger=True)
            #     self.log('val_macro_avg_f1', report['macro avg']['f1-score'], logger=True)
            #     self.log('val_weighted_avg_f1', report['weighted avg']['f1-score'], logger=True)
            #     self.log('val_accuracy', report['accuracy'], logger=True)
        else:
            print("Warning: No validation outputs collected for epoch end")

        # 清空验证步骤输出
        self.validation_step_outputs.clear()

    def validation_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        labels = batch["labels"].reshape(-1, self.n_labels).long()

        last_hidden_state = output.last_hidden_state

        # 提取[CLS]标记
        cls_embedding = last_hidden_state[:, 0, :]

        x = cls_embedding
        for layer in self.hidden_layers:
            x = layer(x)

        logits = self.classifier(x).view(-1, self.n_labels, self.n_classes_per_label)
        probs = torch.softmax(logits, dim=-1)

        prob_class0 = probs[..., :3].sum(dim=-1)  # 形状: (bs, n_labels)
        prob_class1 = probs[..., 3:].sum(dim=-1)  # 形状: (bs, n_labels)

        # 使用类别1的概率作为正类概率
        binary_probs = prob_class1  # 形状: (bs, n_labels)

        # 存储结果用于epoch结束时的指标计算
        self.test_probs.append(binary_probs)
        self.test_targets.append(labels)

        # 更新AUROC指标
        #self.test_auroc.update(binary_probs, labels)
        # 更新指标（自动处理DDP同步）
        self.val_accuracy.update(binary_probs, labels)
        return {'predictions': binary_probs, 'labels': labels}


    def on_validation_epoch_end(self):
        # 自动计算并同步指标
        accuracy = self.val_accuracy.compute()
        self.log('val_overall_accuracy', accuracy, prog_bar=True, logger=True)

        # 重置指标
        self.val_accuracy.reset()

        # 清除其他存储
        self.test_probs.clear()
        self.test_targets.clear()
        self.test_auroc.reset()

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        labels = batch["labels"].reshape(-1, self.n_labels).long()

        last_hidden_state = output.last_hidden_state

        # 提取[CLS]标记
        cls_embedding = last_hidden_state[:, 0, :]

        x = cls_embedding
        for layer in self.hidden_layers:
            x = layer(x)

        logits = self.classifier(x).view(-1, self.n_labels, self.n_classes_per_label)
        probs = torch.softmax(logits, dim=-1)

        prob_class0 = probs[..., :3].sum(dim=-1)  # 形状: (bs, n_labels)
        prob_class1 = probs[..., 3:].sum(dim=-1)  # 形状: (bs, n_labels)

        # 使用类别1的概率作为正类概率
        binary_probs = prob_class1  # 形状: (bs, n_labels)

        # 存储结果用于epoch结束时的指标计算
        self.test_probs.append(binary_probs)
        self.test_targets.append(labels)

        # 更新AUROC指标
        self.test_auroc.update(binary_probs, labels)

        return {'probs': binary_probs, 'labels': labels}


    def on_test_epoch_end(self):
        # 拼接所有batch的结果
        all_probs = torch.cat(self.test_probs, dim=0)  # 形状: (N, n_labels)
        all_targets = torch.cat(self.test_targets, dim=0)  # 形状: (N, n_labels)

        # 转换为numpy用于sklearn指标计算
        probs_np = all_probs.cpu().numpy()
        targets_np = all_targets.cpu().numpy()

        # 计算AUROC（使用torchmetrics）
        auroc = self.test_auroc.compute()
        #self.log("test_auroc", auroc, prog_bar=True)

        # 计算每个标签的AUC（宏平均）
        macro_auc = 0.0
        valid_labels = 0

        # 打印标签分布（调试用）
        #print("\n测试标签分布:")
        for i in range(self.n_labels):
            label_name = self._get_label_name(i)
            unique, counts = np.unique(targets_np[:, i], return_counts=True)
            #print(f"{label_name}: {dict(zip(unique, counts))}")

        # 计算每个标签的AUC
        for i in range(self.n_labels):
            current_target = targets_np[:, i]
            current_probs = probs_np[:, i]

            # 获取标签名称
            label_name = self._get_label_name(i)

            # 检查是否只有一类样本
            if np.unique(current_target).size == 1:
                # 处理单一类别情况
                label_auc = 0.5  # 随机猜测水平
                #print(f"⚠️ {label_name}只有单一类别，AUC设为0.5")
            else:
                # 正常计算AUC
                try:
                    label_auc = roc_auc_score(current_target, current_probs)
                except ValueError:
                    label_auc = 0.5  # 安全回退
                    #print(f"⚠️ {label_name}计算AUC出错，设为0.5")

            # 使用真实标签名称记录结果
            #self.log(f"test_auc_{label_name}", label_auc)

            # 只累加多类别标签的AUC
            if np.unique(current_target).size > 1:
                macro_auc += label_auc
                valid_labels += 1

        # 计算宏平均AUC
        if valid_labels > 0:
            macro_auc /= valid_labels
            #self.log("test_macro_auc", macro_auc, prog_bar=True)
        else:
            #self.log("test_macro_auc", 0.5, prog_bar=True)  # 所有标签都是单一类别时的回退
            #print("⚠️ 所有标签都是单一类别，宏平均AUC设为0.5")
            pass

        # 将概率转换为预测类别（0.5为阈值）
        preds = (probs_np > 0.5).astype(int)

        # 计算精确率、召回率、F1（宏平均）
        precision = precision_score(targets_np, preds, average='macro', zero_division=0)
        recall = recall_score(targets_np, preds, average='macro', zero_division=0)
        f1 = f1_score(targets_np, preds, average='macro', zero_division=0)

        # 计算精确率、召回率、F1（微平均）
        micro_precision = precision_score(targets_np, preds, average='micro', zero_division=0)
        micro_recall = recall_score(targets_np, preds, average='micro', zero_division=0)
        micro_f1 = f1_score(targets_np, preds, average='micro', zero_division=0)

        # 记录所有指标
        #self.log("test_precision_macro", precision)
        #self.log("test_recall_macro", recall)
        #self.log("test_f1_macro", f1)
        #self.log("test_precision_micro", micro_precision)
        #self.log("test_recall_micro", micro_recall)
        #self.log("test_f1_micro", micro_f1)

        # 重置存储
        self.test_probs.clear()
        self.test_targets.clear()
        self.test_auroc.reset()

        # 打印汇总信息
        print("\n测试结果汇总:")
        #print(f"整体AUC: {auroc:.4f}")
        #print(f"宏平均AUC: {macro_auc:.4f}")
        #print(f"宏平均F1: {f1:.4f}")
        #print(f"微平均F1: {micro_f1:.4f}")
        print(f"宏Precision: {precision:.4f}")
        print(f"微Precision: {micro_precision:.4f}")

    # 辅助方法：获取标签名称
    def _get_label_name(self, index):
        """返回指定索引的标签名称"""
        if self.label_names and index < len(self.label_names):
            return self.label_names[index]
        return f"label_{index}"


    def configure_optimizers_saved(self):
        optimizer = AdamW(self.parameters(), lr=self.params['LEARNING_RATE'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

    def configure_optimizers(self):
        # 分层设置学习率
        no_decay = ["bias", "LayerNorm.weight"]

        # BERT参数组（较低学习率）
        bert_params = [
            {
                "params": [p for n, p in self.bert.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.params["WEIGHT_DECAY"],
                "lr": self.params["LEARNING_RATE"] * 0.1,  # BERT学习率更低
            },
            {
                "params": [p for n, p in self.bert.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.params["LEARNING_RATE"] * 0.1,
            },
        ]

        # 分类头参数组（正常学习率）
        classifier_params = [
            {
                "params": [p for n, p in self.named_parameters()
                           if "bert" not in n and not any(nd in n for nd in no_decay)],
                "weight_decay": self.params["WEIGHT_DECAY"],
                "lr": self.params["LEARNING_RATE"],
            },
            {
                "params": [p for n, p in self.named_parameters()
                           if "bert" not in n and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.params["LEARNING_RATE"],
            },
        ]

        optimizer = AdamW(bert_params + classifier_params)

        return optimizer
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
        # 更新调度器
        scheduler.step()

    def predict_step(self, batch, batch_idx):
        # 前向传播
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

        # 获取预测结果
        logits = outputs.logits
        predictions = self._reshape_predictions(logits)
        return predictions
