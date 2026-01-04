import csv
import os
import re

import pandas as pd
import numpy as np
import ast


def extract_values(input_data):
    # 定义所有可能的价值观体现程度
    value_levels = [
        "该价值观毫无体现",
        "该价值观极微弱痕迹",
        "该价值观轻微迹象",
        "该价值观明确表达",
        "该价值观显著表达",
        "该价值观深刻贯彻"
    ]

    # 定义所有价值观维度
    value_dimensions = [
        "博爱", "友善", "权力", "成就", "传统",
        "遵从", "安全", "自主", "刺激", "享乐主义"
    ]

    # 创建空的结果字典，初始化所有列为0
    result_dict = {}
    for dim in value_dimensions:
        for level in value_levels:
            col_name = f"{dim}_{level}"
            result_dict[col_name] = 0

    if pd.isna(input_data) or input_data == "" or input_data is None:
        return pd.DataFrame([result_dict])

    # 增强字符串解析能力
    if isinstance(input_data, str):
        try:
            # 尝试1：直接解析标准格式
            data_dict = ast.literal_eval(input_data)
        except:
            try:
                # 尝试2：替换常见问题字符
                normalized = (
                    input_data
                    .replace("'", "\"")  # 单引号转双引号
                    .replace("，", ",")  # 中文逗号转英文
                    .replace("：", ":")  # 中文冒号转英文
                    .replace("；", ";")  # 中文分号转英文
                    .replace("“", "\"")  # 中文引号转英文
                    .replace("”", "\"")
                    .replace("None", "null")  # Python None -> JSON null
                    .replace("True", "true")  # Python布尔值转换
                    .replace("False", "false")
                )
                data_dict = ast.literal_eval(normalized)
            except:
                try:
                    # 尝试3：处理缺失引号的情况（半结构化修复）
                    if "{" not in input_data:
                        # 尝试包裹成字典格式
                        normalized = "{" + input_data + "}"
                        data_dict = ast.literal_eval(normalized)
                    else:
                        raise ValueError
                except:
                    print(f"无法解析字符串: {input_data[:100]}...")
                    return pd.DataFrame([result_dict])

    # 处理每个价值观维度
    for dim, level_desc in data_dict.items():
        # 只处理已知的价值观维度
        if dim in value_dimensions:
            # 只处理已知的体现程度
            if level_desc in value_levels:
                col_name = f"{dim}_{level_desc}"
                result_dict[col_name] = 1
            else:
                print(f"未知的体现程度: '{level_desc}' 在维度 '{dim}'")

    return pd.DataFrame([result_dict])


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # 替换常见问题字符
    text = text.replace('\r', ' ').replace('\n', ' ')  # 替换换行符
    text = re.sub(r',+', ',', text)  # 减少连续逗号
    text = text.strip()  # 去除首尾空格
    return text


def load_and_process_data(datapath):
    """
    读取目录下的所有CSV文件，合并、处理并创建用于BERT的数据集

    参数:
    datapath (str): 包含CSV文件的目录路径

    返回:
    pd.DataFrame: 处理后的数据集，包含text列和10个价值观列
    """
    # 定义需要提取的列名
    required_columns = [
        '微博正文', 'ip', 'user_gender', 'user_description',
        'user_education', 'user_work', 'user_birthday',
        '博文概括', '凝练后推文_value', '翻译后推文_value',
        '凝练后推文_value_2', '翻译后推文_value_2', '大模型汇总价值观'
    ]

    # 定义字段标识符
    field_identifiers = {
        '微博正文': '[正文]',
        'ip': '[IP]',
        'user_gender': '[性别]',
        'user_description': '[描述]',
        'user_education': '[教育]',
        'user_work': '[职业]',
        'user_birthday': '[生日]',
        '博文概括': '[概括]',
        '凝练后推文_value': '[凝练1]',
        '翻译后推文_value': '[翻译1]',
        '凝练后推文_value_2': '[凝练2]',
        '翻译后推文_value_2': '[翻译2]'
    }

    # 存储所有数据帧的列表
    all_dfs = []

    # 检查目录是否存在
    if not os.path.isdir(datapath):
        raise ValueError(f"目录不存在: {datapath}")

    print(f"开始处理目录: {datapath}")

    # 遍历目录中的所有文件
    file_count = 0
    for filename in os.listdir(datapath):
        if filename.endswith('.csv'):
            filepath = os.path.join(datapath, filename)
            try:
                for encoding in ['utf-8', 'gbk', 'gb18030', 'latin1']:
                    try:
                        # 使用更健壮的csv读取方式
                        df = pd.read_csv(
                            filepath,
                            encoding=encoding,
                            quotechar='"',  # 明确指定引号字符
                            quoting=csv.QUOTE_MINIMAL,  # 仅在必要时使用引号
                            on_bad_lines='warn',  # 遇到错误行时警告而非报错
                            dtype=str  # 所有列先读取为字符串
                        )

                        # 检查并提取所需列
                        available_columns = [col for col in required_columns if col in df.columns]

                        if available_columns:
                            # 只保留需要的列
                            df = df[available_columns]
                            all_dfs.append(df)
                            file_count += 1
                            print(f"成功读取文件: {filename} (编码: {encoding}), 包含 {len(df)} 行")
                            break
                        else:
                            print(f"警告: 文件 {filename} 中未找到任何所需列，跳过")
                            break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"读取文件 {filename} 时出错 (编码: {encoding}): {str(e)}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

    # 合并所有数据帧
    if all_dfs:
        combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)
        print(f"成功合并 {file_count} 个文件，共 {len(combined_df)} 行")

        # 添加缺失的列并用空字符串填充
        for col in required_columns:
            if col not in combined_df.columns:
                combined_df[col] = ""
                print(f"警告: 列 '{col}' 在所有文件中均不存在，已添加为空字符串列")

        # 确保列的顺序一致
        combined_df = combined_df[required_columns]

        # 创建text列：为每个字段添加标识符并拼接
        print("开始创建text列...")
        text_columns = required_columns[:-1]  # 排除最后一列"大模型汇总价值观"

        # 改进3：处理NaN和缺失值
        for col in text_columns:
            if col in combined_df.columns:
                # 转换为字符串并处理NaN
                combined_df[col] = combined_df[col].apply(clean_text)
            else:
                combined_df[col] = ""

        # 改进4：防止标识符与内容混淆
        combined_df['text'] = combined_df.apply(
            lambda row: ' '.join(
                f"{field_identifiers[col]}{row[col].replace(field_identifiers[col], '')}"
                for col in text_columns
            ),
            axis=1
        )

        # 应用extract_values函数处理"大模型汇总价值观"列
        print("开始处理价值观列...")
        values_dfs = []
        for i, row in enumerate(combined_df['大模型汇总价值观']):
            values_df = extract_values(row)
            values_dfs.append(values_df)

            # 进度显示
            if (i + 1) % 1000 == 0:
                print(f"已处理 {i + 1}/{len(combined_df)} 行")

        # 合并所有价值观结果
        values_df = pd.concat(values_dfs, ignore_index=True)
        print("价值观处理完成")

        # 合并text列和价值观列
        final_df = pd.concat([
            combined_df[['text']],
            values_df
        ], axis=1)

        print(f"最终数据集包含 {len(final_df)} 行，列名: {list(final_df.columns)}")

        # 输出示例文本
        print("\n示例文本（包含缺失字段）:")
        for i, text in enumerate(final_df['text'].head(3)):
            print(f"示例 {i + 1}: {text[:200]}...")

        return final_df
    else:
        print("警告: 没有找到任何包含所需列的文件")
        # 创建空数据集，包含所有需要的列
        return pd.DataFrame(columns=['text'] + [
            "博爱", "友善", "权力", "成就", "传统",
            "遵从", "安全", "自主", "刺激", "享乐主义"
        ])


def process_test_data(test_file):
    """
    处理测试集数据，提取特征并创建标签列

    参数:
    test_file (str): 测试集CSV文件路径

    返回:
    pd.DataFrame: 处理后的数据集，包含text列和10个价值观标签列
    """
    # 定义需要提取的列名
    required_columns = [
        '微博正文', 'ip', 'user_gender', 'user_description',
        'user_education', 'user_work', 'user_birthday',
        '博文概括', '凝练后推文_value', '翻译后推文_value',
        '凝练后推文_value_2', '翻译后推文_value_2'
    ]

    # 定义价值观标签列（按训练集顺序）
    value_columns = [
        "博爱", "友善", "权力", "成就", "传统",
        "遵从", "安全", "自主", "刺激", "享乐主义"
    ]

    # 定义字段标识符
    field_identifiers = {
        '微博正文': '[正文]',
        'ip': '[IP]',
        'user_gender': '[性别]',
        'user_description': '[描述]',
        'user_education': '[教育]',
        'user_work': '[职业]',
        'user_birthday': '[生日]',
        '博文概括': '[概括]',
        '凝练后推文_value': '[凝练1]',
        '翻译后推文_value': '[翻译1]',
        '凝练后推文_value_2': '[凝练2]',
        '翻译后推文_value_2': '[翻译2]'
    }

    print(f"开始处理测试文件: {test_file}")

    # 读取测试文件
    for encoding in ['utf-8', 'gbk', 'gb18030', 'latin1']:
        try:
            df = pd.read_csv(
                test_file,
                encoding=encoding,
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
                dtype=str
            )
            print(f"成功读取文件 (编码: {encoding}), 包含 {len(df)} 行")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"读取文件时出错 (编码: {encoding}): {str(e)}")

    # 添加缺失的特征列并用空字符串填充
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""
            print(f"警告: 特征列 '{col}' 不存在，已添加为空字符串列")

    # 添加缺失的标签列并用0填充
    for col in value_columns:
        if col not in df.columns:
            df[col] = 0
            print(f"警告: 标签列 '{col}' 不存在，已添加为0列")

    # 确保特征列的顺序一致
    df = df[required_columns + value_columns]

    # 清理文本数据
    for col in required_columns:
        df[col] = df[col].apply(clean_text)

    # 创建text列
    df['text'] = df.apply(
        lambda row: ' '.join(
            f"{field_identifiers[col]}{row[col].replace(field_identifiers[col], '')}"
            for col in required_columns
        ),
        axis=1
    )

    # 提取价值观标签列（按训练集顺序）
    labels_df = df[value_columns]

    # 合并text列和标签列
    final_df = pd.concat([df[['text']], labels_df], axis=1)

    # 确保标签列顺序正确
    final_df = final_df[['text'] + value_columns]

    print(f"处理完成，最终数据集包含 {len(final_df)} 行")
    print("列结构:", final_df.columns.tolist())

    # 输出示例
    print("\n示例文本:")
    print(final_df['text'].iloc[0][:200] + "...")

    print("\n示例标签:")
    print(final_df[value_columns].iloc[0].to_dict())

    return final_df


# 使用示例
if __name__ == "__main__":
    # 替换为您的实际目录路径
    training_data_path = "../dataset/original_data/"
    processed_data = load_and_process_data(training_data_path)
    # 查看结果
    print("\n处理后的数据集示例（包含缺失字段）:")
    training_data_output_path = "../dataset/training_data.parquet"
    processed_data.to_parquet(training_data_output_path, index=False)
    print(f"\n训练集已保存至: {training_data_output_path}")

    test_file = "../dataset/raw_test_data.csv"  # 替换为实际文件路径
    processed_data = process_test_data(test_file)
    # 保存处理后的数据
    test_data_output_path = "../dataset/test_data.parquet"
    processed_data.to_parquet(test_data_output_path, index=False)
    print(f"\n处理后的数据已保存至: {test_data_output_path}")
