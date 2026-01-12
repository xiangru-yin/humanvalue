import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

class SimpleValueConsistencyCalculator:
    def __init__(self, value_relation_matrix: List[List[int]], columns_dict: Dict[int, str]):
        """
        简化版价值观一致性计算器

        Args:
            value_relation_matrix: 10x10 价值观关系矩阵
                0: 忽略关系
                1: 两个价值观评分越近越真实
                -1: 两个价值观评分越远越真实
            columns_dict: 价值观索引到列名的映射字典
        """
        self.relation_matrix = np.array(value_relation_matrix)
        self.columns_dict = columns_dict
        self.n_values = len(columns_dict)

        # 验证关系矩阵
        assert self.relation_matrix.shape == (
        self.n_values, self.n_values), f"关系矩阵必须是{self.n_values}x{self.n_values}"
        print("价值观关系矩阵:")
        print(self.relation_matrix)

    def calculate_sample_consistency(self, value_scores: np.ndarray) -> float:
        """
        计算单个样本的一致性分数

        Args:
            value_scores: 10个价值观的评分数组 (0-5)

        Returns:
            float: 一致性分数 (0-1, 越高越一致)
        """
        total_consistency = 0
        valid_pairs = 0

        # 遍历所有价值观对
        for i in range(self.n_values):
            for j in range(0, self.n_values):  # 避免重复计算
                relation = self.relation_matrix[i, j]

                if relation == 0:  # 忽略这个关系
                    continue

                # 计算两个价值观的评分差异 (0-5)
                score_diff = abs(value_scores[i] - value_scores[j])

                if relation > 0:  # 越近越好
                    # 差异越小越好，归一化到0-1
                    consistency = abs(relation) * (1.0 - (score_diff / 4.0))
                elif relation < 0:  # 越远越好
                    # 差异越大越好，归一化到0-1
                    consistency = abs(relation) * score_diff / 4.0
                else:
                    continue

                total_consistency += consistency
                valid_pairs += 1

        # 如果有有效关系对，返回平均一致性；否则返回中性值0.5
        return total_consistency / valid_pairs if valid_pairs > 0 else 0.5

    def calculate_data_weights(self, df: pd.DataFrame,
                               power: float = 2.0,
                               min_weight: float = 0.3) -> pd.DataFrame:
        """
        计算所有数据的权重

        Args:
            df: 包含价值观评分的数据框
            power: 权重幂次，用于放大高质量数据的权重
            min_weight: 最小权重，避免数据被完全忽略

        Returns:
            添加了权重的新数据框
        """
        print("开始计算数据权重...")

        # 提取价值观评分
        value_columns = [self.columns_dict[i] for i in range(self.n_values)]
        value_scores = df[value_columns].values

        n_samples = len(df)
        consistency_scores = np.zeros(n_samples)

        # 计算每个样本的一致性
        for idx in range(n_samples):
            scores = value_scores[idx]
            consistency_scores[idx] = self.calculate_sample_consistency(scores)
            for event_true in ['乘客地铁脱鞋，男子将鞋踢出车厢', '俄罗斯空袭乌克兰多个地区', '儿子沉迷手机爸爸逼他连玩17小时', '公公豪掷117万为儿媳买编制被骗', '分娩之痛并非天经地义', '大爷怒斥夜市悬挂日式雨伞', '女儿8万为父看病百万拆迁款给儿子', '女子穿仿制日本军服夜市打闹被行拘', '女生月薪两万辞职考研八次失败', '女童摸狗被咬家长怒斥不赔偿不道歉', '委员称年轻人想躺平更多是调侃', '开放二胎', '教育局回应老师让小学生拿外卖', '春节放鞭炮', '深圳禁摩限电', '父母的衰老会让你感到害怕吗', '电动时代驾驶乐趣在哪儿', '男子工厂打工6年还清65万外债', '男教师被举报猥亵女生后坠亡', '老人疑因男生公交上未让座将其骂哭', '育龄青年为什么不愿生了', '董明珠称大学生去打螺钉没什么不可以', '近9成网友认为苏大开除造谣者合理', '高铁站殴打女童女子系亲生母亲']:
                if event_true in df.loc[idx]['text']:
                    consistency_scores[idx] = 4*1  # 对于valuetalk train数据集内容，默认全对，并增加权重


        # 打印统计信息
        print(f"一致性分数统计:")
        print(f"  平均值: {np.mean(consistency_scores):.4f}")
        print(f"  标准差: {np.std(consistency_scores):.4f}")
        print(f"  最小值: {np.min(consistency_scores):.4f}")
        print(f"  最大值: {np.max(consistency_scores):.4f}")
        amplification = 2.0
        max_weight=5.0
        z_scores = (consistency_scores - np.mean(consistency_scores)) / np.std(consistency_scores)

        # 使用sigmoid函数进行平滑放大
        amplified_scores = 1 / (1 + np.exp(-amplification * z_scores))

        # 映射到权重范围
        weights = min_weight + (max_weight - min_weight) * amplified_scores

        print(f"权重统计:")
        print(f"  平均权重: {np.mean(weights):.4f}")
        print(f"  最小权重: {np.min(weights):.4f}")
        print(f"  最大权重: {np.max(weights):.4f}")

        # 添加到数据框
        result_df = df.copy()
        result_df['consistency_score'] = consistency_scores
        result_df['data_weight'] = weights

        return result_df

    def analyze_extreme_samples(self, df_with_weights: pd.DataFrame,
                                top_k: int = 5) -> None:
        """
        分析最好和最差的样本

        Args:
            df_with_weights: 包含权重信息的数据框
            top_k: 分析的样本数量
        """
        # 最一致的样本
        best_samples = df_with_weights.nlargest(top_k, 'consistency_score')
        print(f"\n最一致的{top_k}个样本:")
        for idx, row in best_samples.iterrows():
            print(f"样本 {idx}: 一致性 = {row['consistency_score']:.4f}, 权重 = {row['data_weight']:.4f}")
            values = [f"{row[self.columns_dict[i]]}" for i in range(self.n_values)]
            print(f"  价值观评分: {', '.join(values)}")

        # 最不一致的样本
        worst_samples = df_with_weights.nsmallest(top_k, 'consistency_score')
        print(f"\n最不一致的{top_k}个样本:")
        for idx, row in worst_samples.iterrows():
            print(f"样本 {idx}: 一致性 = {row['consistency_score']:.4f}, 权重 = {row['data_weight']:.4f}")
            values = [f"{row[self.columns_dict[i]]}" for i in range(self.n_values)]
            print(f"  价值观评分: {', '.join(values)}")


# 使用示例
def calculate_simple_weights(df: pd.DataFrame,
                             value_relation_matrix: List[List[int]],
                             columns_dict: Dict[int, str],
                             power: float = 2.0) -> pd.DataFrame:
    """
    主函数：简化版数据权重计算

    Args:
        df: 原始数据框
        value_relation_matrix: 10x10关系矩阵
        columns_dict: 列名字典
        power: 权重幂次

    Returns:
        添加了权重信息的数据框
    """
    calculator = SimpleValueConsistencyCalculator(value_relation_matrix, columns_dict)

    # 计算权重
    df_with_weights = calculator.calculate_data_weights(df, power)

    # 分析极端样本
    calculator.analyze_extreme_samples(df_with_weights)

    return df_with_weights


# 在训练循环中使用权重的实用函数
def create_weighted_sampler(df_with_weights, dataset):
    """
    创建加权采样器用于训练

    Args:
        df_with_weights: 带权重的数据框
        dataset: 对应的数据集

    Returns:
        加权采样器
    """
    from torch.utils.data import WeightedRandomSampler

    weights = df_with_weights['data_weight'].values
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    return sampler


def apply_sample_weights_to_loss(loss_tensor, weights_tensor, reduction='mean'):
    """
    在损失函数中应用样本权重

    Args:
        loss_tensor: 每个样本的损失张量 [batch_size]
        weights_tensor: 样本权重张量 [batch_size]
        reduction: 减少方式

    Returns:
        加权损失
    """
    if reduction == 'mean':
        return (loss_tensor * weights_tensor).mean()
    elif reduction == 'sum':
        return (loss_tensor * weights_tensor).sum()
    else:
        return loss_tensor * weights_tensor


# 示例：如何在你的训练循环中使用权重
def example_training_usage(df_with_weights, train_dataset, model, params):
    """
    示例：如何在训练中使用计算出的权重
    """
    from torch.utils.data import DataLoader

    # 创建加权采样器
    weighted_sampler = create_weighted_sampler(df_with_weights, train_dataset)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["BATCH_SIZE"],
        sampler=weighted_sampler,
        num_workers=params["NUM_TRAIN_WORKERS"]
    )

    # 在训练步骤中，你可以这样使用权重：
    def weighted_training_step(batch, batch_idx, model, optimizer):
        input_ids, attention_mask, labels, indices = batch  # 假设批次包含样本索引

        # 前向传播
        loss, logits = model(input_ids, attention_mask, labels)

        # 获取当前批次的权重
        batch_weights = torch.tensor(
            df_with_weights.iloc[indices.cpu().numpy()]['data_weight'].values,
            device=loss.device,
            dtype=loss.dtype
        )

        # 计算加权损失
        weighted_loss = apply_sample_weights_to_loss(loss, batch_weights)

        # 反向传播
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        return weighted_loss.item()
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
columns_dict = {
    0:"自主",
    1:"刺激",
    2:"享乐主义",
    3:"成就",
    4:"权力",
    5:"安全",
    6:"传统",
    7:"遵从",
    8:"友善",
    9:"博爱",

}
hanzi_value_dict = {
    "该价值观毫无体现": 0.0,
    "该价值观极微弱痕迹": 0.1,
    "该价值观轻微迹象": 0.3,
    "该价值观明确表达": 0.5,
    "该价值观显著表达": 0.8,
    "该价值观深刻贯彻": 1.0
}
edge=0.5
#df = pd.read_csv("/home/user/for_sync/humanvalue/kfold_confidence_learning_results/cleaned_dataset_kfold.csv", encoding="utf_8_sig")

# 读入目标数据集
df = pd.read_parquet("/home/zxy/for_sync/humanvalue/dataset/train_vt_only_v10.parquet")
original_columns = df.columns

# 去除全无价值观的数据，理论上这种数据应该被价值观过滤去掉了
right_id_list = [True for i in range(len(df))]
for i in range(len(df)):
    for value_name in columns_dict.values():
        find_answer = False
        for num, value_num in enumerate(list(hanzi_value_dict.keys())):
            if df.loc[i, f"{value_name}_{value_num}"] >= edge:
                df.loc[i, f"{value_name}"] = num
                find_answer = True
                break
        if not find_answer:
            # 说明这条数据有问题
            right_id_list[i] = False #抛弃这条数据
df = df[right_id_list]
df = df.reset_index(drop=True)
df_fix = df
#df_fix = df[original_columns]

# 定义各价值观间的一致性关系矩阵和权重
value_relation_matrix = [
    #sd,st,hd,ac,po,se,tr,co,be,un
    [0, 0.5, 0, 0, 0, -1, 0, 0, 0, 0.5],
    [0.5, 0, 0.5, 0, 0, 0, -0.5, -0.5, 0, 0],
    [0, 0.5, 0, 0.5, 0, 0, -1/3, -1/3, -1/3, 0],
    [0, 0, 0.5, 0, 0.5, 0, 0, 0, -1, 0],
    [0, 0, 0, 0.5, 0, 0.5, 0, 0, 0, -1],
    [-1, 0, 0, 0, 0.5, 0, 0.5, 0, 0, 0],
    [0, -1, 0, 0, 0, 0.25, 0, 0.5, 0.25, 0],
    [0, -1, 0, 0, 0, 0.25, 0.5, 0, 0.25, 0],
    [0, 0, 0, -1, 0, 0, 0.25, 0.25, 0, 0.5],
    [0.5, 0, 0, 0, -1, 0, 0, 0, 0.5, 0],
]
# 计算数据权重
df_with_weights = calculate_simple_weights(
    df=df,
    value_relation_matrix=value_relation_matrix,
    columns_dict=columns_dict,
    power=2.0  # 可以调整这个参数
)

# 查看结果
print("\n数据权重统计:")
print(df_with_weights[['consistency_score', 'data_weight']].describe())

# 去除后25%一致性的数据，约0.15左右
df_with_weights = df_with_weights[df_with_weights["consistency_score"] >= 0.15].reset_index(drop=True)

df_with_weights = df_with_weights.drop(columns=list(columns_dict.values()))
df_with_weights = df_with_weights.drop(columns="consistency_score")

# 保存带权重的数据
#df_fix.to_parquet("./dataset/training_data_fix_clean_kfold.parquet", index=False)
df_with_weights.to_parquet('/home/zxy/for_sync/humanvalue/dataset/train_vt_only_v10_weighted.parquet', index=False)

# 在训练中如何使用这些权重
# 1. 在数据加载时使用加权采样
# 2. 在损失计算时应用样本权重