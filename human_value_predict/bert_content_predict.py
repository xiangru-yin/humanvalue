# -*- coding: utf-8 -*-
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm
import traceback
from utils import *

def predict_bert_value_dir_all(dir_path, skip_used=True, tokenizer_list=None, trained_model_list=None, max_length=512,
                               bert_name='deberta',
                               label_columns_list=[
                                   ['思想自主', '行动自主', '刺激', '享乐主义', '成就', '支配权力', '资源权力', '面子',
                                    '个人安全', '社会安全', '传统', '规则遵从', '人际遵从', '谦逊', '友善-关怀',
                                    '友善-可依赖', '博爱-关注', '博爱-大自然', '博爱-宽容', '博爱-客观性']],
                               **kwargs):
    # 集成bert预测数据文件夹内价值观，主要功能为使用不同的预训练bert分析价值观并构造集成结果
    # 输入为待处理的文件夹，输出为预测完成的价值观

    # 如果没有可用的bert模型，则加载默认的
    if not tokenizer_list or not trained_model_list:
        tokenizer = AutoTokenizer.from_pretrained(
            f"..{os.sep}human_value_predict{os.sep}saved_models{os.sep}Deberta_Human_Value_Detector",
            mirror='https://hf-mirror.com', )
        trained_model = AutoModelForSequenceClassification.from_pretrained(
            f"..{os.sep}human_value_predict{os.sep}saved_models{os.sep}Deberta_Human_Value_Detector",
            trust_remote_code=True, mirror='https://hf-mirror.com')
        tokenizer_list = [tokenizer]
        trained_model_list = [trained_model]

    csv_list = get_files_in_folder(dir_path)
    total_len, filter_len = 1, 1
    for data_path in csv_list:
        # 跳过未经上一步处理的和已处理的
        if '_simple' in data_path:
            continue
        if "_lf" not in data_path:
            continue
        try:
            # 对该文件进行处理
            predict_bert_value_file(data_path, tokenizer_list, trained_model_list, skip_used=skip_used,
                                    bert_name=bert_name, max_length=max_length, label_columns_list=label_columns_list,
                                    **kwargs)

        except Exception as e:
            traceback.print_exc()
            print(f"失败，文件：{data_path}，原因：{e}")
    # example_text = '#A cold wave will arrive before New Year’s Eve#'
    return dir_path


# 使用示例

def predict_bert_value_file(file_path, tokenizer_list, trained_model_list, skip_used=True, max_length=512,
                            bert_name='deberta',
                            label_columns_list=[
                                ['思想自主', '行动自主', '刺激', '享乐主义', '成就', '支配权力', '资源权力',
                                 '面子', '个人安全', '社会安全', '传统', '规则遵从', '人际遵从', '谦逊',
                                 '友善-关怀', '友善-可依赖', '博爱-关注', '博爱-大自然', '博爱-宽容',
                                 '博爱-客观性']],
                            **kwargs
                            ):
    # 用不同的bert处理该文件，主要功能为使用初始化好的bert预测价值观，并构造集成结果
    # 输入为csv文件位置，输出暂无意义
    input_file = file_path
    output_file = input_file.replace(".csv", f"_simple.csv")

    df = read_process_csv(input_file, output_file, skip_if=skip_used)

    # 筛选有概括结果的数据，以免属性不足导致无法预测
    df = df[df["凝练后推文"] != ""]
    df = df.reset_index(drop=True)

    # 导出数据
    content_list = list(df['凝练后推文'])

    # 获取事件tag
    df['tag_en'] = df['翻译后推文'].apply(lambda x: get_tag(x))
    tag = df['tag_en']
    tag_list = list(tag)

    # 进行整体的bert价值观预测
    content_value_list = bert_value_predict_simple(content_list, tag_list, max_length,
                                              batch_size=kwargs.get('batch_size', 1), THRESHOLD=0.25,
                                              tokenizer_list=tokenizer_list, trained_model_list=trained_model_list,
                                              label_columns_list=label_columns_list)
    # 将结果保存下来
    df['凝练后推文_value'] = content_value_list
    df.to_csv(output_file, encoding='utf_8_sig')

    # 筛选有翻译结果的，以免数据属性不足导致无法运行
    df = df[df["翻译后推文"] != ""]

    # 导出数据
    content_list = list(df['翻译后推文'])

    # 提取tag
    tag = df['tag_en']
    tag_list = list(tag)

    # 分句进行价值观预测
    comment_value = bert_value_predict_split(content_list, tag_list, max_length,
                                             batch_size=kwargs.get('batch_size', 1), THRESHOLD=0.25,
                                             tokenizer_list=tokenizer_list, trained_model_list=trained_model_list,
                                             label_columns_list=label_columns_list)
    # 保存句子的平均价值观
    df['翻译后推文_value'] = comment_value



    # 生成集成结果
    if True:
        # 处理各属性列
        df.loc[:, '凝练后推文_value'] = df.loc[:, '凝练后推文_value'].apply(lambda x: str(x))
        df.loc[:, '翻译后推文_value'] = df.loc[:, '翻译后推文_value'].apply(lambda x: str(x))

        # 去除无结果的
        df = df[df['凝练后推文_value'] != ""]
        df = df[df['翻译后推文_value'] != ""]

        # 转化为文字
        df.loc[:, "凝练后推文_value_2"] = df.loc[:, "凝练后推文_value"].apply(
            lambda x: str(ast.literal_eval(x)[-1]) if len(x) > 5 else "[]")
        df.loc[:, "凝练后推文_value"] = df.loc[:, "凝练后推文_value"].apply(
            lambda x: str(ast.literal_eval(x)[0]) if len(x) > 5 else "[]")
        df.loc[:, "翻译后推文_value_2"] = df.loc[:, "翻译后推文_value"].apply(
            lambda x: str(np.array(ast.literal_eval(x))[:, -1, :].tolist()) if len(x) > 5 else "[[]]")
        df.loc[:, "翻译后推文_value"] = df.loc[:, "翻译后推文_value"].apply(
            lambda x: str(np.array(ast.literal_eval(x))[:, 0, :].tolist()) if len(x) > 7 else "[[]]")

        # 增加text属性列
        df = df.reset_index(drop=True)
        df['text'] = ""
        gender_dict = {
            "男": "m", "女": "f", "f": "f", "m": "m"
        }
        df = df.fillna("")

        # 确定各个属性名字
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

    df_copy = df.copy()
    if "text" in df_copy.columns:
        df_copy = df_copy.drop(columns="text")

    # 转化为bert的简短形式
    # 如果要用大模型来预测，可以不转化

    # 将数字用文字表达
    df = process_num_to_str(df)
    df = df.fillna("")

    # 生成集成结果
    df['text'] = df.apply(
        lambda row: '[SEP]'.join(
            f"{field_identifiers[col]}{row[col].replace(field_identifiers[col], '').strip()}"
            for col in field_identifiers
        ),
        axis=1
    )

    # 保存文件
    df = pd.merge(left=df_copy, right=df[["id", "text"]], on="id", how="right")
    df.to_csv(output_file.replace('.csv', '_split.csv'), encoding='utf_8_sig')


def bert_value_predict(comment_list: list, tag_list: list, max_length=512, batch_size=1, THRESHOLD=0.25,
                       tokenizer_list=[], trained_model_list=[], label_columns_list=[]) -> list:
    # 通用的bert预测模块，实现对所给文字列进行预测
    # 输入为待分析文本列表，输出为分析结果列表

    # 价值观列
    LABEL_COLUMNS = ['思想自主', '行动自主', '刺激', '享乐主义', '成就', '支配权力', '资源权力', '面子', '个人安全',
                     '社会安全', '传统', '规则遵从', '人际遵从', '谦逊', '友善-关怀', '友善-可依赖', '博爱-关注',
                     '博爱-大自然', '博爱-宽容', '博爱-客观性']

    print(f"Predictions:")

    # 逐个文本进行分析
    comment_value = []
    for i in tqdm(range(len(comment_list))):
        # 读取文本主体和文本tag
        comment = comment_list[i]
        example_text = tag_list[i]
        one_prediction = []

        # 逐个模型进行分析
        for model_i in range(len(tokenizer_list)):
            # 读取该模型的配件
            tokenizer = tokenizer_list[model_i]
            trained_model = trained_model_list[model_i]
            label_columns = label_columns_list[model_i]

            # 编码文本
            if "roberta" in tokenizer.name_or_path:
                # Roberta只支持165的max_token
                encoding = tokenizer.encode_plus(
                    example_text + " " + comment,
                    add_special_tokens=True,
                    max_length=min(165, max_length),
                    return_token_type_ids=False,
                    padding="max_length",
                    return_attention_mask=True,
                    return_tensors='pt',
                    truncation=True,

                )
            else:
                encoding = tokenizer.encode_plus(
                    example_text + " " + comment,
                    add_special_tokens=True,
                    max_length=max_length,
                    return_token_type_ids=False,
                    padding="max_length",
                    return_attention_mask=True,
                    return_tensors='pt',
                    truncation=True,

                )

            # 模型分析
            with torch.no_grad():
                test_prediction = trained_model(encoding["input_ids"].to(trained_model.device),
                                                encoding["attention_mask"].to(trained_model.device))
                test_prediction = test_prediction[1] if isinstance(test_prediction, tuple) else test_prediction[
                    "output"]
                test_prediction = test_prediction.flatten().cpu().numpy()

            # 解析分类结果，转化为二分类
            prediction_result = {}
            for label, prediction in zip(label_columns, test_prediction):
                if prediction < THRESHOLD:
                    pass
                # print(f"{label}: {prediction}")
                prediction_result[label] = float(prediction)

            # 转化为统一格式
            prediction_result = [prediction_result.get(label_i, 0) for label_i in LABEL_COLUMNS]
            one_prediction.append(prediction_result)
        if len(one_prediction) == 1:
            # 兼容单条的情况
            one_prediction = one_prediction[0]
        # 保存结果
        comment_value.append(one_prediction)
    return comment_value


def bert_value_predict_simple(*args, **kwargs) -> list:
    # 直接预测即可
    return bert_value_predict(*args, **kwargs)


def bert_value_predict_split(comment_list: list, tag_list: list, *args, **kwargs) -> list:
    # 对文本进行分句后进行分类
    comment_list_extend = []
    tag_list_extend = []
    comment_index_list = []
    comment_result = []
    for i, comment in enumerate(comment_list):
        comment_result.append([])
        if "[SEP]" in comment:
            split_flag = "[SEP]"
        else:
            split_flag = ". "

        for split_comment in comment.split(split_flag):
            if len(split_comment.strip()) == 0:
                continue
            else:
                comment_list_extend.append(split_comment)
                comment_index_list.append(i)
                tag_list_extend.append(tag_list[i])

    # 交给bert进行分类
    comment_value = bert_value_predict(comment_list_extend, tag_list_extend, *args, **kwargs)

    # 整理分类结果
    for i, index in enumerate(comment_index_list):
        comment_result[index].append(comment_value[i])

    return comment_result
