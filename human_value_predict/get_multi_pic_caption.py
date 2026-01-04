# encoding:utf-8
from tqdm import tqdm
import traceback
from basic_config import *
from utils import *


# 需要实现的功能：将图片组描述为博文
def llm_prompt_pic_describe(blog="", pic_length=0):
    prompt = f"""请你参考其博文内容，用类似微博博文的风格简短话语描述和概括该博文的{pic_length}张附属图片
请不要以前景背景的方式来介绍，而是以类似博文的方式整体介绍图片。请不要分开介绍各个图片，而是一起概括汇总。
输出形式如下dict形式: 
{{'图片概述':"用简短的中文描述和概括所有图片的含义，采用微博的风格"}}
不要回复其他的内容，只回复你的分析结果dict
示例输出：
{{'图片概述': "新闻报道现场洪水滔天，受难人数众多。有许多网友回复，为难民祈福"}}
{{'图片概述': "自拍美照"}}
{{'图片概述': "马克龙在国旗下发表讲话。而伊朗核设施目前正在建设"}}

参考博文如下：
{blog}
"""

    return prompt


def llm_describe_data_dir(dir_path, skip_used=True):
    # 基于大模型的筛选和数据增强，主要功能为调用大模型对数据进行价值观筛选，并对通过筛选的生成总结和翻译
    # 输入为文件夹地址，输出暂无用途
    csv_list = get_files_in_folder(dir_path)
    path_need = []
    for data_path in csv_list:
        # 跳过处理过的和未经上一步处理的
        if '_lf' in data_path:
            continue
        if "filter" not in data_path:
            continue
        try:
            #data_flen, data_olen = llm_trans_file(data_path, skip_used=skip_used)
            #total_len, filter_len = total_len + data_olen, filter_len + data_flen
            path_need.append(data_path)
        except Exception as e:
            print(f"过滤失败，文件：{data_path}，原因：{e}")

    from joblib import Parallel, delayed
    if use_parallel_if:
        Parallel(n_jobs=min(parallel_max_num, os.cpu_count() - 3, len(path_need)))(
            delayed(llm_describe_pic_file)(path_need[i], skip_used=skip_used) for i in tqdm(range(len(path_need))))
    else:
        for i in tqdm(range(len(path_need))):
            llm_describe_pic_file(path_need[i])

    return dir_path


def llm_describe_pic_file(input_file, skip_used=True):
    # 大语言模型对csv文件进行多模态增强，主要功能为调用大模型对图片进行理解和概括
    # 输入为文件位置，输出为概括前数据量和概括后数据量
    local_dir = os.sep.join(input_file.split(os.sep)[:-1])
    pic_dir = local_dir + os.sep + 'images'

    # input_file = "valuetalk_process_filter.csv"
    result_file = input_file.replace('.csv', '.csv') # 不需要额外后缀
    df = read_process_csv(input_file, output_file=result_file, skip_if=skip_used)

    multi_modal_index_list = list(df[df["微博图片url"]!=""].index)

    # 增加新的属性列
    if '图片概述' not in df.columns:
        df['图片概述'] = ""
    df = df.fillna("")

    # 开始逐个处理运行，由于难以说明图片和微博间的所属关系，无法并行运行

    for i in tqdm(multi_modal_index_list):
        # 取出该数据
        index_i = i
        # 如果已被处理过了，则跳过
        if df.iloc[index_i]['图片概述'] != "":
            continue
        if df.iloc[index_i]['微博图片url'] == "":
            continue
        # 取出该批次的文本数据
        content = df.iloc[index_i]['content']
        pic_list = df.iloc[index_i]["微博图片url"].split(",")
        pic_length = len(pic_list)
        if not use_local_pic_if:
            # 使用图片url进行传递，无需本地下载
            pass
        else:
            # 使用本地图片
            if pic_length <= 0:
                continue
            elif pic_length == 1:
                pic_list = []
                pic_path = pic_dir + os.sep + f"{df.iloc[index_i]['id']}.jpg"
                base64_image = encode_image(pic_path)
                if base64_image is not None:
                    pic_list = [f"data:image/jpg;base64,{base64_image}"]
                else:
                    # 文件无法读取，跳过
                    continue
            else:
                # 多张图片，依据尾号识别
                pic_list = []
                for pic_i in range(pic_length):
                    if pic_i == 0:
                        pic_path = pic_dir + os.sep + f"{df.iloc[index_i]['id']}.jpg"
                    else:
                        pic_path = pic_dir + os.sep + f"{df.iloc[index_i]['id']}-{pic_i}.jpg"
                    base64_image = encode_image(pic_path)
                    if base64_image is not None:
                        pic_list += [f"data:image/jpg;base64,{base64_image}"]
                    else:
                        continue
        # 构造批次文本提示词
        if len(pic_list) <= 0:
            # 实际没有图片
            continue
        content_string = str(content)
        prompt = llm_prompt_pic_describe(blog=content_string, pic_length=pic_length)
        # prompt = llm_prompt(blog=content)

        # 获取回应
        response = ""
        for times in range(3):
            try:
                if i == 0:
                    print(prompt)
                response = api_response(user_prompt=prompt, multi_modal=multi_modal, pic_url=pic_list)
                if i == 0:
                    print(response)
                # 处理大模型回应
                if "json" in response:
                    response = response.replace("```json", "")
                    response = response.replace("```", "")
                if "python" in response:
                    response = response.replace("```python", "")
                    response = response.replace("```", "")
                if "”}" in response:
                    response = response.replace("”}", "\"}")
                if "{”" in response:
                    response = response.replace("{”", "{\"")

                # 检查是否达标
                result = eval(response)
                assert type(result) == dict

                assert "图片概述" in result


                df.loc[index_i, '图片概述'] = result['图片概述']
                df.loc[index_i, 'content'] += result['图片概述']

                break
            except Exception as e:
                print(f"失败，重新尝试第{i + 1}次，回复为{response}，error:{e}")
                traceback.print_exc()
        # 保存结果，防止丢失
        if i % 5 == 0:
            df.to_csv(result_file, encoding='utf_8_sig')

    # 只保留有概括结果的数据，即有价值观的数据
    df = df.reset_index(drop=True)
    df.to_csv(result_file, encoding='utf_8_sig')

    return df

if __name__ == '__main__':
    llm_describe_pic_file("D:\Pycharm\watch_system\human_value_predict\human_value_data\外交部\外交部回应演员星星失联.csv")

