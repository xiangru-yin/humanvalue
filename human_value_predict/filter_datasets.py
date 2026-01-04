import json
from utils import *
from basic_config import *



def read_keypoint():
    # 从关键词json文件中读取关键词列表，形成不同类别的关键词字典
    # 关键词文件位置由ord_file， weibo_file来指定


    # 读取json文件，导出关键词
    with open(ord_file, 'r', encoding='utf-8') as f:
        ord_key = json.load(f)
    with open(weibo_file, 'r', encoding='utf-8') as f:
        weibo_key = json.load(f)
    key_list = []
    #从导出的关键词json数据中
    key_all = ord_key
    # 往ord_key中增加weibo_key，并全部导出来
    for key in weibo_key.keys():
        data_k = weibo_key[key]
        for attribute in data_k.keys():
            data_att = data_k[attribute]
            key_all[key][attribute] += data_att
            key_list += key_all[key][attribute]
    key_list = list(set(key_list))

    return key_list


def filter_data_dir(dir_path):
    # 文件夹级别的价值观过滤模块，主要功能为对文件夹下未处理的csv文件继续处理，并为其带上_filter的后缀
    # 输入为文件夹地址，输出暂无用途


    csv_list = get_files_in_folder(dir_path)

    # 开始处理，并统计处理情况
    total_len, filter_len = 1, 1
    for data_path in csv_list:
        if '_filter' in data_path: # 不要处理过的
            continue
        if "enrich" not in data_path: # 可能有问题的
            print('价值观未检测到enrich后缀，请检查是否正确')
        try:
            data_flen, data_olen = filter_data(data_path) # 进行文件级别的价值观过滤
            total_len, filter_len = total_len + data_olen, filter_len + data_flen # 更新统计结果
        except Exception as e:
            print(f"过滤失败，文件：{data_path}，原因：{e}")
    print(f"filter_rate:{filter_len/total_len}") # 汇报过滤情况
    print(f"filtered_len:{filter_len}") # 汇报过滤情况
    return dir_path

def filter_data(data_path):
    # 文件级别的过滤处理，主要功能为读入指定csv文件和关键词表，去除所有不包含关键词的数据
    # 输入为待过滤的csv文件地址
    # 输出为过滤后数据大小，用于统计过滤情况

    def value_if(x):
        # 判断是否包含至少一个关键词
        for key in value_key:
            if key in x:
                return True
        return False

    # 读入关键词表
    value_key = read_keypoint()

    # 确定输入和输出文件路径
    input_file = data_path
    output_file = input_file.replace('.csv', '_filter.csv')

    # 读入和预处理csv
    df = read_process_csv(input_file, output_file)
    original_len = len(df)

    # 关键词筛选，只留下包含至少一个关键词的数据
    df_if = df['content'].apply(value_if)
    df_copy = df[df_if]

    # 如果过滤后数据过少，为防止样本太少导致统计误差，放弃过滤
    if len(df_copy) <= 30:
        print(f"放弃过滤，数量太少，filter_len:{len(df_copy)}，filter_rate:{len(df_copy)/original_len}, 文件{input_file}")
    else:
        df = df_copy

    # 导出文件为utf_8_sig格式csv文件
    #print(f"filtered_len:{len(df)}")
    df.to_csv(output_file, index=False, encoding='utf_8_sig')
    pass

    # 返回处理后数据大小，处理前数据大小
    return len(df), original_len

if __name__ == "__main__":
    filter_data_dir("D:\Pycharm\watch_system\human_value_predict\human_value_data\wuhan_example_test\\12306回应大妈高铁车厢跳舞喧哗")