import csv
import os
import re

import pandas as pd
import numpy as np
import ast

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 定义的各属性对应的简写名称
field_identifiers = {

        '凝练后推文_value': '凝练一',
        '凝练后推文_value_2': '凝练二',
        '翻译后推文_value': '翻译一',
        '翻译后推文_value_2': '翻译二',
        '博文概括': '概括',

        'user_gender': '性别',

        "user_followers": "粉丝",
        'user_following': '关注',
        'user_work': '职业',
        'user_birthday': '生日',
        "user_education": '教育',
        "user_authentication": "身份",
        "user_verified_reason": "认证",

        'ip': 'IP',
        '用户昵称': "昵称",
        'user_description': '描述',

        '话题': "话题",
        '微博正文': '正文',
    }

def extract_values(input_data):
    # 解析保存的大模型标注价值观数据

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
        print("故障")
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
                        print(f"无法解析字符串: {input_data[:100]}...")
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
    # 清理文本

    if pd.isna(text):
        return ""
    text = str(text)
    # 替换常见问题字符
    text = text.replace('\r', ' ').replace('\n', ' ')  # 替换换行符
    text = re.sub(r',+', ',', text)  # 减少连续逗号
    text = text.strip()  # 去除首尾空格
    return text

def get_files_in_folder(folder_path):
    # 获取所有的csv数据样本，并返回csv_list
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file_path in files:
            if "csv" in file_path:
                file_list.append(os.path.join(root, file_path))
    return file_list

def load_and_process_data(datapath):
    """
    读取目录下的所有CSV文件，合并、处理并创建用于BERT的数据集

    参数:
    datapath (str): 包含CSV文件的目录路径

    返回:
    pd.DataFrame: 处理后的数据集，包含text列和10个价值观列
    """
    # 定义需要提取的列名
    # 定义字段标识符

    required_columns = list(field_identifiers.keys()) + ["大模型汇总价值观", "所属事件"]

    # 存储所有数据帧的列表
    all_dfs = []

    # 检查目录是否存在
    if not os.path.isdir(datapath):
        raise ValueError(f"目录不存在: {datapath}")

    print(f"开始处理目录: {datapath}")

    # 遍历目录中的所有文件
    file_count = 0
    for filename in get_files_in_folder(datapath):
        if filename.endswith('.csv'):
            filepath = filename
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

        combined_df = process_num_to_str(combined_df)

        # 创建text列：为每个字段添加标识符并拼接
        print("开始创建text列...")
        text_columns = required_columns[:-2]  # 排除最后一列"大模型汇总价值观"

        # 改进3：处理NaN和缺失值
        for col in text_columns:
            if col in combined_df.columns:
                # 转换为字符串并处理NaN
                combined_df[col] = combined_df[col].apply(clean_text)
            else:
                combined_df[col] = ""

        # 改进4：防止标识符与内容混淆
        combined_df['text'] = combined_df.apply(
            lambda row: '[SEP]'.join(
                f"{field_identifiers[col]}{row[col].replace(field_identifiers[col], '').strip()}"
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
            combined_df[['text', "所属事件"]],
            values_df
        ], axis=1)

        final_df = final_df.drop(columns="所属事件")
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



def convert_number_to_text(number):
    """
    将数字转换为中文文本描述
    大于1的数字按标准量级划分，小于1的小数四舍五入保留第一位小数并只显示其小数部分
    """
    if not isinstance(number, (int, float)):
        try:
            number = float(number)
        except:
            return number
    # 处理负数
    if number < 0:
        return "负" + convert_number_to_text(-1 * number)

    # 处理小于1的小数
    if 0 < number < 1:
        if int(round(number, 1)*10) < 10:
            return str(int(round(number, 1)*10))
        else:
            #将10变为十
            return "十"

    # 处理0
    if number == 0:
        return "0"

    # 处理1
    if number == 1:
        return "1"

    # 处理大于1的数字
    if 1 < number < 10:
        return f"{int(round(number))}"

    if 10 <= number < 100:
        return "十"

    if 100 <= number < 1000:
        return "百"

    if 1000 <= number < 10000:
        return "千"

    if 10000 <= number < 100000:
        return "万"

    if 100000 <= number < 1000000:
        return "十万"

    if 1000000 <= number < 10000000:
        return "百万"

    if 10000000 <= number < 100000000:
        return "千万"

    if 100000000 <= number < 1000000000:
        return "亿"

    if 1000000000 <= number < 10000000000:
        return "十亿"

    # 更大的数字
    return "极大数"
def process_num_to_str(df:pd.DataFrame):
    # 处理数字列，转变为文字

    # 依次将所有的数字转化为文本
    if "user_followers" in df.columns:
        df.loc[:, 'user_followers'] = df.loc[:, 'user_followers'].apply(lambda x:convert_number_to_text(x))

    if "user_following" in df.columns:
        df.loc[:, 'user_following'] = df.loc[:, 'user_following'].apply(lambda x:convert_number_to_text(x))

    if "凝练后推文_value" in df.columns:
        df.loc[:, '凝练后推文_value'] = df.loc[:, '凝练后推文_value'].apply(lambda x:str([convert_number_to_text(num) for num in ast.literal_eval(x)]))
        df.loc[:, '凝练后推文_value'] = df.loc[:, '凝练后推文_value'].apply(
            lambda x: x.replace(" ", "").replace("\"", "").replace("'", "").replace(",", "").replace("[", "").replace(
                "]", ""))

    if "凝练后推文_value_2" in df.columns:
        df.loc[:, '凝练后推文_value_2'] = df.loc[:, '凝练后推文_value_2'].apply(lambda x:str([convert_number_to_text(num) for num in ast.literal_eval(x)]))
        df.loc[:, '凝练后推文_value_2'] = df.loc[:, '凝练后推文_value_2'].apply(
            lambda x: x.replace(" ", "").replace("\"", "").replace("'", "").replace(",", "").replace("[", "").replace(
                "]", ""))



    if "翻译后推文_value" in df.columns:
        # 使用句子级价值观结果的平均值作为输入
        df.loc[:, '翻译后推文_value'] = df.loc[:, '翻译后推文_value'].apply(
            lambda x: np.array(ast.literal_eval(x)).mean(axis=0).tolist())
        df.loc[:, '翻译后推文_value'] = df.loc[:, '翻译后推文_value'].apply(
            lambda x: str([convert_number_to_text(num) for num in x]) if type(x)==list else str([]))
        df.loc[:, '翻译后推文_value'] = df.loc[:, '翻译后推文_value'].apply(
            lambda x: x.replace(" ", "").replace("\"", "").replace("'", "").replace(",", "").replace("[", "").replace("]", ""))
    if "翻译后推文_value_2" in df.columns:
        # 使用句子级价值观结果的平均值作为输入
        df.loc[:, '翻译后推文_value_2'] = df.loc[:, '翻译后推文_value_2'].apply(
            lambda x: np.array(ast.literal_eval(x)).mean(axis=0).tolist())
        df.loc[:, '翻译后推文_value_2'] = df.loc[:, '翻译后推文_value_2'].apply(
            lambda x: str([convert_number_to_text(num) for num in x]) if type(x) == list else str([]))
        df.loc[:, '翻译后推文_value_2'] = df.loc[:, '翻译后推文_value_2'].apply(
            lambda x: x.replace(" ", "").replace("\"", "").replace("'", "").replace(",", "").replace("[", "").replace(
                "]", ""))

    df = df.fillna("")

    return df
# 使用示例
if __name__ == "__main__":
    # 替换为您的实际目录路径
    training_data_path = "../dataset/original_data/"
    processed_data = load_and_process_data(training_data_path)
    #processed_data.to_csv("../dataset/wuhan_test.csv", index=False, encoding='utf_8_sig')
    # 查看结果
    print("\n处理后的数据集示例（包含缺失字段）:")
    training_data_output_path = "../dataset/wuhan_train_weighted.parquet"
    processed_data.to_parquet(training_data_output_path, index=False)
    print(f"\n训练集已保存至: {training_data_output_path}")

