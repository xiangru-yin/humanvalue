# encoding:utf-8
from tqdm import tqdm
import traceback

from basic_config import *
from utils import *



# 功能：去除白开水，翻译成英文，凝练为一两句话，
def llm_prompt_judge(blog="", time="", user_info="None", blog_info='None'):
    prompt = f"""请你逐个判断所给出的微博博文是否有价值观，你的任务如下：
1.判断所给出的微博博文是否包含或表达了价值观，若没有表达，如博文是白开水，或博文只是在进行事实称述，没有包含或表达价值观，或博文是广告等无关文本，则在"价值观_if"中回复"False"并跳过下面的任务，开始分析下一个博文，否则回复"True"，
2.用推特语境下的一两句中文凝练浓缩表达博文，尽可能保留原博文所含有的价值观，回复于"中文推文概述"中 
3.输出翻译结果，输出形式如下dict形式: 
{{
0: {{'价值观_if':"是否包含价值观",'中文推文概述':"凝练后的中文推文"}},
1: {{'价值观_if':"是否包含价值观",'中文推文概述':"凝练后的中文推文"}},
2: {{'价值观_if':"是否包含价值观",'中文推文概述':"凝练后的中文推文"}}
}}
不要回复其他的内容，只回复你的分析结果dict
示例输出：
{{ 
0: {{'价值观_if': "True",'中文推文概述': "一段概述"}}, 
1: {{'价值观_if': "False",'中文推文概述': ""}}, 
2: {{'价值观_if': "False",'中文推文概述': ""}}, 
3: {{'价值观_if': "True",'中文推文概述': "一段概述"}}, 
4: {{'价值观_if': "False",'中文推文概述': ""}}, 
5: {{'价值观_if': "True",'中文推文概述': "一段概述"}}, 
}}
输入的目标微博博文字典如下，请依次分析：
{blog}
"""

    return prompt
def llm_prompt_trans_simple(blog="", time="", user_info="None", blog_info='None'):
    prompt = f"""你是一个翻译机器人，你需要逐行翻译给出的中文微博博文到英文推文，你很擅长将中文语境下的微博博文翻译到英文语境下的推文，你会逐行翻译，保持翻译前后的行数相同，你会保留原本的格式，不要翻译为list或dict格式。
输入示例：
4:翻译示例文本1
5:翻译示例文本2
输出示例：
4:translated example text 1
5:translated example text 2
你要翻译的文本如下：
{blog}
"""
    return prompt




def llm_filter_data_dir(dir_path, skip_used=True):
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
            delayed(llm_trans_file)(path_need[i], skip_used=skip_used) for i in tqdm(range(len(path_need))))
    else:
        for i in tqdm(range(len(path_need))):
            llm_trans_file(path_need[i], skip_used=skip_used)

    return dir_path
def llm_trans_file(input_file, skip_used=True):
    # 大语言模型对csv文件进行筛选和增强，主要功能为调用大模型对数据进行价值观筛选，并对通过筛选的生成总结和翻译
    # 输入为文件位置，输出为筛选前数据量和筛选后数据量

    # input_file = "valuetalk_process_filter.csv"
    result_file = input_file.replace('.csv', '_lf.csv')
    df = read_process_csv(input_file, output_file=result_file, skip_if=skip_used)
    original_len = len(df)
    # 依据句号对原文进行切分，划分不同句子
    df['content'] = df['content'].apply(lambda x: x.replace('\n', '。').replace('。。', '。'))
    df['content_split'] = df['content'].apply(lambda x: x.split("。"))

    if "llm_filter_if" not in df.columns:
        df['llm_filter_if'] = ""

    # 增加新的属性列
    if '凝练后推文' not in df.columns:
        df[['llm_filter_if', '凝练后推文', '翻译后推文', '博文概括']] = ""
    df = df.fillna("")

    # 计算分批运行次数
    total_step = (len(df) // batch) + 1

    # 开始分批运行
    for i in tqdm(range(total_step)):
        # 取出该批次数据
        start = i * batch
        end = min(len(df) - 1, (i + 1) * batch)

        # 终止条件
        if end <= start:
            continue

        # 如果该批次最后一条已被处理过了，则跳过
        if df.iloc[end]['llm_filter_if'] != "":
            continue
        # 取出该批次的文本数据
        content = df.iloc[start:end]['content'].tolist()
        content = {k: content[k] for k in range(len(content))}

        # 构造批次文本提示词
        content_string = "{\n"
        for k in range(len(content)):
            content_string += f"{k}:'{content[k]}',\n"
        content_string += "}\n}"
        prompt = llm_prompt_judge(blog=content_string)
        # prompt = llm_prompt(blog=content)

        # 获取回应
        response = ""
        for times in range(3):
            try:
                if i == 0:
                    print(prompt)
                response = api_response(user_prompt=prompt, multi_modal=multi_modal)
                if i == 0:
                    print(response)
                # 处理大模型回应
                if "json" in response:
                    response = response.replace("```json", "")
                    response = response.replace("```", "")
                if "python" in response:
                    response = response.replace("```python", "")
                    response = response.replace("```", "")

                # 检查是否达标
                result = eval(response)
                assert type(result) == dict
                for key in content:
                    assert key in result
                assert len(result) == len(content)

                # 将结果填入批次内各位置
                for t in range(len(result)):
                    result_t = result[t]
                    if result_t['价值观_if'] == "false" or result_t['价值观_if'] == 'False':
                        # 不具有价值观的，无需后续处理
                        df.loc[i * batch + t, 'llm_filter_if'] = False
                        continue
                    # 填入概括结果
                    df.loc[i * batch + t, 'llm_filter_if'] = True
                    df.loc[i * batch + t, '凝练后推文'] = result_t['中文推文概述']

                break
            except Exception as e:
                print(f"失败，重新尝试第{i + 1}次，回复为{response}，error:{e}")
                traceback.print_exc()
        # 保存结果，防止丢失
        if i % 5 == 0:
            df.to_csv(result_file, encoding='utf_8_sig')

    # 只保留有概括结果的数据，即有价值观的数据
    df = df[df['凝练后推文'] != ""]
    df = df.reset_index(drop=True)
    df.to_csv(result_file, encoding='utf_8_sig')

    N_NUM = batch//3 # 由于平均句子数的存在，建议降低批量大小


    # 开始分批运行
    total_step = (len(df) // N_NUM) + 1
    for i in tqdm(range(total_step)):
        # 确定该批次的范围
        start = i * N_NUM
        end = min(len(df) - 1, (i + 1) * N_NUM)

        # 结束条件
        if end <= start:
            continue

        # 跳过已经处理了的
        if df.iloc[end]['翻译后推文'] != "":
            continue

        # 取出该批次数据
        content = df.iloc[start:end]['content_split'].tolist()
        for now_i in range(len(content)):
            content[now_i].append(df.iloc[start + now_i]['凝练后推文'])

        # 构建该批次数据的文本表达
        content_belong_dict = {}
        content_string = ""
        now_count = 0
        for k in range(len(content)):
            for w in range(len(content[k])):
                content_string += f"{now_count}:{content[k][w]}\n"
                content_belong_dict[now_count] = k
                now_count += 1
        content_string += ""

        # 构造提示词
        prompt = llm_prompt_trans_simple(blog=content_string)

        # 请求大模型回复
        response = ""
        for times in range(4):
            try:
                if i == 0:
                    print(prompt)
                response = api_response(user_prompt=prompt, multi_modal=multi_modal)
                if i == 0:
                    print(response)

                # 处理大模型回复
                try:
                    # 尽管不允许回复为list，但还是很多时候都会回复为list
                    # 回复为list时，有可能由于翻译文本中的'，"或者]而导致解析失败
                    response = ast.literal_eval(response)
                except:
                    response = response.split('\n')
                result = {}

                # 尝试解析大模型回复
                idx = 0
                for response_part in response:
                    response_part = response_part.replace("：", ":")
                    content_idx = ":".join(response_part.split(':')[1:])
                    if len(response_part) <= 0:
                        continue
                    try:
                        # 寻找数字开头
                        idx = int(response_part.split(':')[0])
                        result[idx] = content_idx
                    except:
                        # 没有数字开头，那应该是原文太长或原文中就有换行，导致输出到另外一行了
                        result[idx] += content_idx
                        pass

                # 检查回复完整性
                for key in content_belong_dict:
                    assert key in result, print(content_string)
                assert len(result) == len(content_belong_dict)

                # 将数据放于批次内各位置
                for t in range(len(result)):
                    result_t = result[t]

                    # 翻译的是原文的句子拆分
                    if t + 1 in result and content_belong_dict[t + 1] == content_belong_dict[t]:

                        if df.loc[i * N_NUM + content_belong_dict[t], '翻译后推文'] != '':
                            # 合并结果，以[SEP]作为句子间分隔符
                            df.loc[i * N_NUM + content_belong_dict[t], '翻译后推文'] = df.loc[i * N_NUM +
                                                                                              content_belong_dict[
                                                                                                  t], '翻译后推文'] + '[SEP]' + result_t
                        else:
                            # 空结果，不划分为一句
                            df.loc[i * N_NUM + content_belong_dict[t], '翻译后推文'] += result_t

                    else:
                        # 翻译的是概括部分
                        df.loc[i * N_NUM + content_belong_dict[t], '博文概括'] = df.loc[
                            i * N_NUM + content_belong_dict[t], '凝练后推文']
                        df.loc[i * N_NUM + content_belong_dict[t], '凝练后推文'] = result_t

                break
            except Exception as e:
                print(f"失败，重新尝试第{i + 1}次，回复为{response}，error:{e}")
                traceback.print_exc()
        # 不断保存
        if i % 5 == 0:
            df.to_csv(result_file, encoding='utf_8_sig')
    df.to_csv(result_file, encoding='utf_8_sig')

    # 汇报筛选结果
    print(f"filter_rate:{len(df) / (original_len+1)}")
    return len(df), original_len

if __name__ == '__main__':
    llm_filter_data_dir("D:\Pycharm\watch_system\human_value_predict\human_value_data\wuhan_example_test\\12306回应大妈高铁车厢跳舞喧哗")

