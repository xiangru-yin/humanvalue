import pandas as pd
import numpy as np
import re

from basic_config import *
import openai

import base64

# 遍历文件夹的csv
def get_files_in_folder(folder_path):
    # 获取所有的csv数据样本，并返回csv_list
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file_path in files:
            if "csv" in file_path:
                file_list.append(os.path.join(root, file_path))
    return file_list

# 读取csv文件
def read_process_csv(input_file, output_file="", skip_if=False):
    # 检查有无处理过
    if os.path.exists(output_file) and skip_if:
        # 之前处理过
        input_file = output_file
        print("继续处理：" + input_file)


    # 读入文件
    try:
        df = pd.read_csv(input_file, encoding='utf_8_sig')
    except:
        df = pd.read_csv(input_file, encoding='gbk')

    # 文件预处理，统一转化为content和time数据列，避免影响原始数据

    if 'content' not in df.columns:
        df['content'] = df['微博正文']
    if 'time' not in df.columns:
        df['time'] = df['发布时间']
    df = df.drop_duplicates(subset="id")
    df = df.fillna("")

    return df

# 获取事件tag
def get_tag(content):
    if "#" not in content:
        return content
    if content.count("#") >= 2:
        # 找到第一个 # 的位置
        first_hash_index = content.find('#')

        # 找到最后一个 # 的位置
        last_hash_index = content.rfind('#')

        assert last_hash_index > first_hash_index

        return content[first_hash_index:last_hash_index + 1]
    else:
        first_hash_index = content.find('#')
        content = content[first_hash_index:]
        last_hash_index = content.find('[SEP]')
        if last_hash_index != -1:
            return content[:last_hash_index]
        else:
            last_hash_index = content.find(' ')
            if last_hash_index != -1:
                return content[:last_hash_index] + "#"
            else:
                return content

# 转化数字为文本
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
        if int(round(number, 1) * 10) < 10:
            return str(int(round(number, 1) * 10))
        else:
            # 将10变为十
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

# 将数字用文字描述
def process_num_to_str(df: pd.DataFrame):
    # 处理数字列，转变为文字
    field_identifiers = {
        '微博正文': '[正文]',
        '话题': "[话题]",
        'ip': '[IP]',
        '用户昵称': "[昵称]",
        'user_gender': '[性别]',
        'user_description': '[描述]',
        "user_followers": "[粉丝]",
        'user_following': '[关注]',
        'user_work': '[职业]',
        'user_birthday': '[生日]',
        "user_education": '[教育]',
        "user_authentication": "[身份]",
        "user_verified_reason": "[认证]",
        # '博文概括': '[概括]',
        # '凝练后推文_value': '[凝练1]',
        # '翻译后推文_value': '[翻译1]',
        # '凝练后推文_value_2': '[凝练2]',

        # '翻译后推文_value_2': '[翻译2]'
    }
    # 逐步处理各个列
    if "user_followers" in df.columns:
        df.loc[:, 'user_followers'] = df.loc[:, 'user_followers'].apply(lambda x: convert_number_to_text(x))

    if "user_following" in df.columns:
        df.loc[:, 'user_following'] = df.loc[:, 'user_following'].apply(lambda x: convert_number_to_text(x))

    if "凝练后推文_value" in df.columns:
        df.loc[:, '凝练后推文_value'] = df.loc[:, '凝练后推文_value'].apply(
            lambda x: str([convert_number_to_text(num) for num in ast.literal_eval(x)]))
        df.loc[:, '凝练后推文_value'] = df.loc[:, '凝练后推文_value'].apply(
            lambda x: x.replace(" ", "").replace("\"", "").replace("'", "").replace(",", "").replace("[", "").replace(
                "]", ""))

    if "凝练后推文_value_2" in df.columns:
        df.loc[:, '凝练后推文_value_2'] = df.loc[:, '凝练后推文_value_2'].apply(
            lambda x: str([convert_number_to_text(num) for num in ast.literal_eval(x)]))
        df.loc[:, '凝练后推文_value_2'] = df.loc[:, '凝练后推文_value_2'].apply(
            lambda x: x.replace(" ", "").replace("\"", "").replace("'", "").replace(",", "").replace("[", "").replace(
                "]", ""))

    if "翻译后推文_value" in df.columns:
        df.loc[:, '翻译后推文_value'] = df.loc[:, '翻译后推文_value'].apply(
            lambda x: np.array(ast.literal_eval(x)).mean(axis=0).tolist())
        df.loc[:, '翻译后推文_value'] = df.loc[:, '翻译后推文_value'].apply(
            lambda x: str([convert_number_to_text(num) for num in x]) if type(x) == list else str([]))
        # 鉴于debert词表中没有多位数，将数字压缩
        df.loc[:, '翻译后推文_value'] = df.loc[:, '翻译后推文_value'].apply(
            lambda x: x.replace(" ", "").replace("\"", "").replace("'", "").replace(",", "").replace("[", "").replace(
                "]", ""))
    if "翻译后推文_value_2" in df.columns:
        df.loc[:, '翻译后推文_value_2'] = df.loc[:, '翻译后推文_value_2'].apply(
            lambda x: np.array(ast.literal_eval(x)).mean(axis=0).tolist())
        df.loc[:, '翻译后推文_value_2'] = df.loc[:, '翻译后推文_value_2'].apply(
            lambda x: str([convert_number_to_text(num) for num in x]) if type(x) == list else str([]))
        df.loc[:, '翻译后推文_value_2'] = df.loc[:, '翻译后推文_value_2'].apply(
            lambda x: x.replace(" ", "").replace("\"", "").replace("'", "").replace(",", "").replace("[", "").replace(
                "]", ""))

    df = df.fillna("")

    return df

# openai 风格的api访问
def api_response(messages=None, system_prompt="", user_prompt="",  pic_url:str="", video_url='',
                 multi_modal=multi_modal, content_model=content_model, multi_model=multi_model, **kwargs):
    # 依据prompt和参数，调用GPT
    # 设置API key

    if not multi_modal or pic_url == "":
        if content_model=='deepseek-chat':
            client = openai.OpenAI(
                # 下面两个参数的默认值来自环境变量，可以不加
                api_key=api_key,
                base_url=api_url,
            )
        elif content_model=='deepseek-reasoner':
            client = openai.OpenAI(
                # 下面两个参数的默认值来自环境变量，可以不加
                # api_key = "sk-uy7jtk0gZwdvA3TdjsEtO9bUGttHUZ2PcrsQrx5ZAu73ySTz",
                api_key=api_key,
                base_url=api_url,
            )
        elif content_model == 'local':
            client = openai.OpenAI(
                # 下面两个参数的默认值来自环境变量，可以不加
                api_key=api_key,  # deepseek api
                base_url=api_url,
            )
        elif content_model == 'gpt-4o':
            client = openai.OpenAI(
                # 下面两个参数的默认值来自环境变量，可以不加
                api_key=api_key,
                base_url=api_url,
            )
        elif "qwen" in content_model:
            client = openai.OpenAI(
                # 下面两个参数的默认值来自环境变量，可以不加
                api_key=api_key,
                base_url=api_url,
            )
        else:
            print("No such model!")
    else:
        if multi_model == 'gpt-4o':
            client = openai.OpenAI(
                # 下面两个参数的默认值来自环境变量，可以不加
                api_key=api_key,
                base_url=api_url,
            )
        elif multi_model == "qwen-omni-turbo":
            client = openai.OpenAI(
                # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
                api_key=api_key,
                base_url=api_url,
            )
        elif multi_model == 'local':
            client = openai.OpenAI(
                # 下面两个参数的默认值来自环境变量，可以不加
                api_key=api_key,
                base_url=api_url,
            )
        else:
            print("没有这个模型")
            raise

    if not multi_modal or pic_url == "":
        if content_model == 'local':
            completion = client.chat.completions.create(
                model="Qwen/Qwen3-8B-Base",
                messages=[
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt,
                     }
                ],
                # stream=True,
                **kwargs
            )
        elif "deepseek" in content_model:
            if len(system_prompt) > 1:
                completion = client.chat.completions.create(
                    model=content_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt,
                         "tool_calls": [
                             {
                                 "id": "string",
                                 "type": "function",
                                 "function": {
                                     "name": "string",
                                     "arguments": "string"
                                 }
                             }
                         ]
                         }
                    ],

                    # stream=True,
                    **kwargs
                )
            else:
                completion = client.chat.completions.create(
                    model=content_model,
                    messages=[
                        # {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt,
                         "tool_calls": [
                             {
                                 "id": "string",
                                 "type": "function",
                                 "function": {
                                     "name": "string",
                                     "arguments": "string"
                                 }
                             }
                         ]
                         }
                    ],

                    # stream=True,
                    **kwargs
                )
        elif "gpt" in content_model:
            user_content = [{
                "type": "text",
                "text": user_prompt
            }]
            sys_content = [{
                "type": "text",
                "text": system_prompt
            }]
            if len(system_prompt) > 1:
                completion = client.chat.completions.create(
                    # model="qwen-omni-turbo",
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": sys_content},
                        {"role": "user", "content": user_content}
                    ],
                    **kwargs
                )
                return completion.choices[0].message.content
            else:
                completion = client.chat.completions.create(
                    # model="qwen-omni-turbo",
                    model="gpt-4o",
                    messages=[
                        # {"role": "system", "content": system_prompt},
                        {"role": "user",
                         "content": user_content}
                    ],
                    **kwargs
                )
        elif "qwen" in content_model:

            user_content = [{
                "type": "text",
                "text": user_prompt
            }]
            sys_content = [{
                "type": "text",
                "text": system_prompt
            }]
            if len(system_prompt) > 1:
                completion = client.chat.completions.create(
                    model=content_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",
                         "content": user_content}
                    ],
                    stream=True,
                    stream_options={
                        "include_usage": True
                    },
                    **kwargs
                )
            else:
                completion = client.chat.completions.create(
                    model="qwen-omni-turbo",
                    messages=[
                        # {"role": "system", "content": system_prompt},
                        {"role": "user",
                         "content": user_content}
                    ],
                    stream=True,
                    stream_options={
                        "include_usage": True
                    },
                    **kwargs
                )
            result = ""
            for chunk in completion:
                if chunk.choices:
                    one_char = chunk.choices[0].delta.content
                    if type(one_char) == str:
                        result += one_char
            return result
        return completion.choices[0].message.content
    else:
        user_content = [{
            "type": "text",
            "text": user_prompt
        }]
        if type(pic_url) == str and len(pic_url) >= 1:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": pic_url
                }
            })
            print("已启用图片多模态")

        if type(pic_url) == list and len(pic_url) >= 1:
            user_content += [{"type": "image_url", "image_url": {"url": pic_i}} for pic_i in pic_url]
            print("已启用多图片多模态")


        if type(video_url) == str and len(video_url) >= 1:
            user_content.append({
                "type": "video_url",
                "video_url": {
                    "url": video_url
                }
            })
            print("已启用视频多模态")
        if multi_model == 'gpt-4o':
            for i, one_content in enumerate(user_content):
                if "video_url" in one_content:
                    del user_content[i]
                    break
            completion = client.chat.completions.create(
                # model="qwen-omni-turbo",
                model="gpt-4o",
                messages=[
                    # {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": user_content}
                ],
                **kwargs
            )
            return completion.choices[0].message.content
        elif multi_model == "qwen-omni-turbo":
            if type(video_url) == str and len(video_url) >= 1:
                # 不支持混合模态
                user_content = [{
                    "type": "text",
                    "text": user_prompt
                }, {
                    "type": "video_url",
                    "video_url": {
                        "url": video_url
                    }
                }]
            completion = client.chat.completions.create(
                model="qwen-omni-turbo",
                messages=[
                    # {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": user_content}
                ],
                stream=True,
                stream_options={
                    "include_usage": True
                },
                **kwargs
            )
            result = ""
            for chunk in completion:
                if chunk.choices:
                    one_char = chunk.choices[0].delta.content
                    if type(one_char) == str:
                        result += one_char
            return result
        elif multi_model == "local":
            completion = client.chat.completions.create(
                model="string",
                messages=[
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                # stream=True,
                **kwargs
            )
            return completion.choices[0].message.content
        else:
            print("没有这个模型")
            return ""

# 对字典进行去重
def drop_duplicate_value(original_dict):
    #对字典去重

    # 转换为DataFrame
    df = pd.DataFrame(original_dict)
    df_columns = df.columns
    # 按照title去重（保留第一个出现的记录）
    deduplicated_df = df.drop_duplicates(subset=['title'], keep='first')

    # 转换回字典格式
    result_dict = {

    }
    for col in df_columns:
        result_dict[col] = deduplicated_df[col].tolist(),

    return result_dict

# 去除事件tag
def remove_event_tags(text):
    """
    去除博文中的事件TAG（格式如 #事件名称#）
    保留TAG外的其他内容（包括其他非事件话题标签）
    """
    # 匹配 #事件名称# 格式的TAG（中文/英文/数字/下划线）
    pattern = r'#[^#]+?#'
    cleaned_text = re.sub(pattern, '', text)
    #pattern = r'【[^#]+?】'
    #cleaned_text = re.sub(pattern, '', cleaned_text)
    # 去除微博视频和大括号
    cleaned_text = cleaned_text.replace("的微博视频", '').replace("【】", '').replace("【：", '【').replace("【，", '【')
    return cleaned_text.strip()

# 编码函数： 将本地文件转换为 Base64 编码的字符串
def encode_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    else:
        return None

