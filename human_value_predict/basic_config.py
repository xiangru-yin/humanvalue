import os
import ast

# 关键词筛选部分
ord_file = f"..{os.sep}human_value_predict{os.sep}ord_keypoint.json"
weibo_file = f"..{os.sep}human_value_predict{os.sep}weibo_keypoint.json"

# 大语言模型筛选和增强部分
#api_key = "sk-810524e0a49742faaaf4ecd11fbe766a" # ds api
#api_key = "sk-uy7jtk0gZwdvA3TdjsEtO9bUGttHUZ2PcrsQrx5ZAu73ySTz" #gpt
api_key = "sk-3ee5b27fad274c898b0c7e0442e7d3df" # qwen api
#api_url = "https://api.deepseek.com" # ds url
#api_url = "https://xiaoai.plus/v1" # gpt url
api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1" # qwen url

use_parallel_if = True
parallel_max_num = 20
batch = 10 # 多少个数据打包成一批一起处理，需注意，对于翻译来说，其实际为batch*平均句子数
use_local_pic_if = True

# content_model = 'deepseek-chat'
#content_model = 'deepseek-chat' # 文本模型
content_model = 'qwen2.5-vl-7b-instruct'
# content_model = 'local'
# multi_model = 'gpt-4o'
multi_model = "qwen2.5-vl-7b-instruct"
# multi_model = "local"
multi_modal = True

# bert集成分析部分

model_config_dic = {
             'cyx_model':{
                 "checkpoint_path":"D:\Pycharm\watch_system\human_value_predict\humanvalue_api\checkpoints\last-0.796.ckpt",# 训练好的
                 "model_name":'IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese', # hugging_face 网址，但这个需要重新训练
                 "local_model_path":'saved_models/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese'
             },
             "bert_model_list":[
                 {
                     'model_path':f"..{os.sep}human_value_predict{os.sep}saved_models{os.sep}Deberta_Human_Value_Detector",
                     'tokenizer_path': f'..{os.sep}human_value_predict{os.sep}saved_models{os.sep}Deberta_Human_Value_Detector',
                     'model_name':'tum-nlp/Deberta_Human_Value_Detector',#hugging_face 网址
                     'label_columns': ['思想自主', '行动自主', '刺激', '享乐主义', '成就', '支配权力', '资源权力',
                                           '面子', '个人安全', '社会安全', '传统', '规则遵从', '人际遵从', '谦逊',
                                           '友善-关怀', '友善-可依赖', '博爱-关注', '博爱-大自然', '博爱-宽容',
                                           '博爱-客观性'],

                 },
                {
                     'model_path':f'..{os.sep}human_value_predict{os.sep}saved_models{os.sep}danschr{os.sep}roberta-large-BS_16-EPOCHS_5-LR_5e-05-ACC_GRAD_2-MAX_LENGTH_165{os.sep}HCV-371-danschr-roberta-large-BS_16-EPOCHS_8-LR_5e-05-ACC_GRAD_2-MAX_LENGTH_165-BS_8-LR_2e-05-HL_None-DROPOUT_None-SL_None.ckpt',
                     'tokenizer_path':f'..{os.sep}human_value_predict{os.sep}saved_models{os.sep}danschr{os.sep}roberta-large-BS_16-EPOCHS_5-LR_5e-05-ACC_GRAD_2-MAX_LENGTH_165',
                     'model_name':f'roberta-large-BS_16-EPOCHS_5-LR_5e-05-ACC_GRAD_2-MAX_LENGTH_165',#hugging_face 网址
                     'params_path':f'..{os.sep}human_value_predict{os.sep}saved_models{os.sep}danschr{os.sep}roberta-large-BS_16-EPOCHS_5-LR_5e-05-ACC_GRAD_2-MAX_LENGTH_165{os.sep}HCV-371_PARAMS.pkl',
                     'label_columns':['思想自主','行动自主','刺激','享乐主义','成就','支配权力','资源权力','面子','个人安全','社会安全','传统','规则遵从','人际遵从','谦逊','友善-关怀','友善-可依赖','博爱-关注','博爱-大自然','博爱-宽容','博爱-客观性'],
                 },
             ],
             'llm_model':{
                 'model_name':'local_model',
                 'api_key':'sk-212eea9b28d9425d935f9b4815c1a5ab',
                 'multi_modal':False,
             }
         }



# k均值聚类和可视化部分
enhance_performence = True
llm_or_bert = True
sharp_internel_weight = 0.5
one_file_with_multi_event = False

# 最后的输出格式接口，方便调整
def trans_data_screen(data: dict):
    # 转化数据格式，主要功能为将数据转化为呈交给服务器的新数据格式，并去除额外的属性
    # 输入为原始数据字典，输出为转化后的数据字典
    # 专门一个函数来适配不同的数据格式，大多数情况下只需要修改这里

    result = {}

    # 转化帖子列表部分
    result['typical_posts'] = [{
            "title": data['person']['title'][0][data_i],
            "url": data['person']['url'][0][data_i],
            "source": "weibo",
            "datetime": data['person']['time'][0][data_i].split('+')[0]+":00",
            "heat": data['person']['sharp'][0][data_i],
            "autonomy": data['person']['value'][0][data_i][0],
            "stimulus": data['person']['value'][0][data_i][1],
            "fraternity": data['person']['value'][0][data_i][9],
            "friendliness": data['person']['value'][0][data_i][8],
            "compliance": data['person']['value'][0][data_i][7],
            "tradition": data['person']['value'][0][data_i][6],
            "security": data['person']['value'][0][data_i][5],
            "authority": data['person']['value'][0][data_i][4],
            "achievement": data['person']['value'][0][data_i][3],
            "hedonism": data['person']['value'][0][data_i][2]
        } for data_i in range(len(data['person']['title'][0]))], #LABEL_COLUMNS = ['自主', '刺激', '享乐主义', '成就', '权力', '安全', '传统', '遵从', '友善', '博爱']

    # 转化群体组成部分
    data['population_compose'] = ast.literal_eval(str(data['human_pie']).replace('subData', 'population_values'))
    data['population_compose'] = [{**data_p} for data_p in data['population_compose']]
    result["population_composition"] = data['population_compose']

    result["event"] = data["event"]

    return result
