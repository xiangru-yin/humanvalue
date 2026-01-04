import sys
import os

import sys
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录（Service目录）
parent_dir = os.path.dirname(current_dir)
# 添加父目录到sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(f"当前工作目录: {os.getcwd()}")
print(f"添加的父目录: {parent_dir}")
print(f"当前sys.path: {sys.path}")
from basic_config import *
from utils import *
from filter_datasets import filter_data_dir
from LLM_filter_datasets_qwen import llm_filter_data_dir
#from DHV_content_simple import predict_bert_value_dir_simple
#from DHV_content_split import predict_bert_value_dir_split
from bert_content_predict import predict_bert_value_dir_all
from LLM_ensemble import get_value_ensemble_dir, get_value_ensemble_dir_multi
from k_cluster import get_dir_group_value
import shutil
import os
import shutil
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from BertFineTunerPl import BertFineTunerPl
from get_multi_pic_caption import llm_describe_data_dir
import random
import torch
import transformers
import bert_ensemble_cyx as cyx_api
import os
random.seed(2025)
human_value_model_yxr_dir = {
    'bert_model_list':[],
    'llm_model':{},
    'params':{
        'device':'cpu',
        'batch_size':1,
        'local_model_path':f'..{os.sep}human_value_predict{os.sep}saved_models{os.sep}Deberta_Human_Value_Detector',
        'outside_model':'local',
        'max_token_count':512,
        'num_workers':0,
    }
}

# 全局变量



def init(checkpoint_path=f"checkpoints{os.sep}value_finest.ckpt",
         batch_size=1,
         max_token_count=512,
         num_workers=0,
         device=None,
         model_name=f'IDEA-CCNL{os.sep}Erlangshen-DeBERTa-v2-320M-Chinese',
         use_local_model=True,
         local_model_path=f'..{os.sep}human_value_predict{os.sep}saved_models{os.sep}Deberta_Human_Value_Detector',
         model_config_dic = model_config_dic):
    """
    初始化价值观预测器
    """
    print("="*50)
    print("初始化价值观预测器...")
    print("="*50)

    # 设置随机种子
    random_seed = 2025
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # 设备
    if device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        _device = device
    print(f"使用设备: {_device}")

    # 参数
    human_value_model_yxr_dir['params']['batch_size'] = batch_size
    human_value_model_yxr_dir['params']['max_token_count'] = max_token_count
    human_value_model_yxr_dir['params']['num_workers'] = num_workers



    # tokenizer model
    for bert_config_dict in model_config_dic['bert_model_list']:
        model_name = bert_config_dict['model_name']
        tokenizers_path = bert_config_dict.get('tokenizer_path', '')
        local_model_path = bert_config_dict.get('model_path', '')
        print(f"加载tokenizer: {model_name}")
        if use_local_model and local_model_path:
            _tokenizer = AutoTokenizer.from_pretrained(
                tokenizers_path,
                local_files_only=True,
                mirror='https://hf-mirror.com',
                trust_remote_code=True,
                cache_dir=f'..{os.sep}human_value_predict{os.sep}saved_models',
            )
        else:
            try:
                _tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    mirror='https://hf-mirror.com',
                    trust_remote_code=True,
                    use_fast=True
                )
            except Exception:
                from transformers import BertTokenizerFast
                _tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese", mirror='https://hf-mirror.com',)

        print("Tokenizer加载成功！")

        # 模型
        print(f"加载模型权重: {checkpoint_path}")
        if 'Deberta' in model_name:
            _model = AutoModelForSequenceClassification.from_pretrained(
                f"..{os.sep}human_value_predict{os.sep}saved_models{os.sep}Deberta_Human_Value_Detector", trust_remote_code=True,
                local_files_only=True, cache_dir=f'..{os.sep}human_value_predict{os.sep}saved_models', mirror='https://hf-mirror.com',)

        elif 'roberta' in model_name:
            import pickle
            with open(bert_config_dict['params_path'], 'rb') as f:
                PARAMS = pickle.load(f)
            _model = BertFineTunerPl.load_from_checkpoint(
                local_model_path,
                params=PARAMS,
                label_columns=bert_config_dict['label_columns'],
                n_classes=len(bert_config_dict['label_columns']),
                strict=False  # 关键参数
            )
        elif 'roberta' in model_name and False:
            import pickle
            with open(bert_config_dict['params_path'], 'rb') as f:
                PARAMS = pickle.load(f)

            # 直接创建HF模型而不是PL模型
            from transformers import RobertaForSequenceClassification
            from transformers import RobertaForSequenceClassification, RobertaConfig
            # 创建配置
            config = RobertaConfig.from_pretrained(
                model_name,
                mirror='https://hf-mirror.com',
                num_labels=len(bert_config_dict['label_columns'])
            )

            # 创建模型
            _model = RobertaForSequenceClassification(config)

            # 加载PL检查点的权重
            checkpoint = torch.load(local_model_path, map_location='cpu', weights_only=True)

            # 转换键名
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('model.'):
                    state_dict[k[6:]] = v

            # 加载权重
            _model.load_state_dict(state_dict, strict=False)

            del checkpoint
            torch.cuda.empty_cache()
        else:
            _model = AutoModelForSequenceClassification.from_pretrained(local_model_path, trust_remote_code=True, mirror='https://hf-mirror.com')
        _model.eval()
        _model.to(_device)
        human_value_model_yxr_dir['bert_model_list'].append({**bert_config_dict, 'tokenizer':_tokenizer, 'model':_model, 'label_columns':bert_config_dict['label_columns']})

        print("模型加载成功！")



    print("="*50)
    print("初始化完成！")
    print("="*50)

    cyx_api.init()


def forward(event_name="",
            event_data_csv_path="", dst_dir=f'..{os.sep}human_value_predict{os.sep}human_value_data{os.sep}',
            event_image_dir="", skip_used=True, filter_dict=dict(), user_if=False):
    """
    进行预测
    """
    result_data = {}
    global human_value_model_yxr_dir
    dir_path = merge_copy_directory(event_data_csv_path, dst_dir)
    # 进行关键词筛选
    dir_path = filter_data_dir(dir_path)
    # 进行多模态增强
    dir_path = llm_describe_data_dir(dir_path)
    # 进行大语言模型筛选和增强
    dir_path = llm_filter_data_dir(dir_path)
    tokenizer_list = []
    trained_model_list = []
    label_columns_list = []
    for model_dict in human_value_model_yxr_dir['bert_model_list']:
        tokenizer_list.append(model_dict['tokenizer'])
        trained_model_list.append(model_dict['model'])
        label_columns_list.append(model_dict['label_columns'])
    # 进行集成bert分析
    dir_path = predict_bert_value_dir_all(dir_path, skip_used=False, tokenizer_list=tokenizer_list, trained_model_list=trained_model_list, label_columns_list=label_columns_list)
    # 进行bert汇总分析
    dir_path = cyx_api.get_value_ensemble_dir_bert(dir_path, skip_used=skip_used)
    #dir_path = get_value_ensemble_dir_multi(dir_path, skip_used=False)
    # 进行群体分析和可视化
    result, result_data = get_dir_group_value(dir_path)


    return result_data



def merge_copy_directory(origin_dir, data_dir):
    """
    将origin_dir文件夹及其内容合并复制到data_dir下，形成data_dir{os.sep}origin_dir{os.sep}xxxx结构
    如果目标文件夹已存在，则合并内容（相同文件名会覆盖）

    :param origin_dir: 要复制的源文件夹路径
    :param data_dir: 目标文件夹路径
    """
    try:
        # 确保目标目录存在
        os.makedirs(data_dir, exist_ok=True)

        # 获取源文件夹名称
        origin_dir_name = os.path.basename(os.path.normpath(origin_dir))

        # 构造目标路径
        dest_path = os.path.join(data_dir, origin_dir_name)

        # 如果目标路径不存在，直接复制整个目录
        if not os.path.exists(dest_path):
            shutil.copytree(origin_dir, dest_path)
            print(f"成功将 {origin_dir} 复制到 {dest_path}")
            return dest_path

        # 如果目标路径存在，则遍历并合并每个文件子目录
        print(f"目标路径 {dest_path} 已存在，开始合并...")

        for root, dirs, files in os.walk(origin_dir):
            # 计算相对路径
            rel_path = os.path.relpath(root, origin_dir)
            dest_root = os.path.join(dest_path, rel_path)

            # 确保目标子目录存在
            os.makedirs(dest_root, exist_ok=True)

            # 复制所有文件
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dest_root, file)

                if os.path.exists(dst_file):
                    #print(f"覆盖文件: {dst_file}")
                    pass
                else:
                    shutil.copy2(src_file, dst_file)  # copy2会保留元数据


        print(f"成功将 {origin_dir} 合并到 {dest_path}")
        return dest_path

    except Exception as e:
        print(f"复制过程中发生错误: {e}")
        return

if __name__ == "__main__":
    #predict_value_dir('wb_评论')
    #get_value_dir('wb_用户文件_学生')
    init()
    forward(event_name="",
            event_data_csv_path=f"..{os.sep}human_value_data{os.sep}外交部",
            skip_used=True)