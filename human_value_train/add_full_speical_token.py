from transformers import AutoTokenizer, AutoModel
import torch
import os

model_name = "/home/user/for_sync/humanvalue/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese"

# 加载模型和tokenizer
print("正在加载模型和tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

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
    '博文概括': '[概括]',
    '凝练后推文_value': '[凝练1]',
    '翻译后推文_value': '[翻译1]',
    '凝练后推文_value_2': '[凝练2]',
    '翻译后推文_value_2': '[翻译2]'
}

# 定义特殊token与近义词的映射
synonym_mapping = {
    '[正文]': ['正文', '内容', '文本', '文章'],
    '[话题]': ['话题', '主题', '议题', '讨论'],
    '[IP]': ['IP', '地址', '网络', '位置'],
    '[昵称]': ['昵称', '用户名', '称呼', '代号'],
    '[性别]': ['性别', '男女', '性', '性别特征'],
    '[描述]': ['描述', '介绍', '说明', '简述'],
    '[粉丝]': ['粉丝', '关注者', '追随者', '听众'],
    '[关注]': ['关注', '关心', '注意', '留意'],
    '[职业]': ['职业', '工作', '职位', '行业'],
    '[生日]': ['生日', '诞辰', '出生', '生辰'],
    '[教育]': ['教育', '学历', '学习', '学校'],
    '[身份]': ['身份', '地位', '角色', '标识'],
    '[认证]': ['认证', '验证', '确认', '证明'],
    '[概括]': ['概括', '总结', '归纳', '概述'],
    '[凝练1]': ['凝练', '精简', '简洁', '简明'],
    '[翻译1]': ['翻译', '译文', '转换', '译'],
    '[凝练2]': ['凝练', '精简', '简洁', '简明'],
    '[翻译2]': ['翻译', '译文', '转换', '译']
}

# 定义新的特殊token
special_tokens_to_add = list(field_identifiers.values())
special_tokens_to_add = list(set(special_tokens_to_add))

print(f"原始tokenizer词汇表大小: {len(tokenizer)}")

# 检查tokenizer是否已经有这些token
existing_tokens = []
new_tokens = []
for token in special_tokens_to_add:
    if token in tokenizer.get_vocab():
        existing_tokens.append(token)
    else:
        new_tokens.append(token)

print(f"已存在的token: {existing_tokens}")
print(f"需要添加的新token: {new_tokens}")

# 只添加不存在的token
if new_tokens:
    tokenizer.add_special_tokens({
        "additional_special_tokens": new_tokens,
    })
    print(f"添加了 {len(new_tokens)} 个新token")
else:
    print("所有token都已存在，无需添加")

print(f"更新后tokenizer词汇表大小: {len(tokenizer)}")

# 调整模型embedding大小（如果需要添加新token）
if new_tokens:
    model.resize_token_embeddings(len(tokenizer))
    print(f"模型embedding大小已调整为: {model.get_input_embeddings().weight.shape[0]}")

# 获取embedding层
input_embeddings = model.get_input_embeddings()
old_vocab_size = input_embeddings.weight.shape[0] - len(new_tokens)


def find_best_synonym_embedding(special_token, synonym_mapping, tokenizer, embeddings, old_vocab_size):
    """
    为特殊token找到最佳近义词的embedding
    """
    if special_token not in synonym_mapping:
        # 如果没有定义近义词映射，使用平均embedding
        return embeddings[:old_vocab_size].mean(dim=0, keepdim=True)

    synonyms = synonym_mapping[special_token]

    # 尝试找到在词汇表中存在的近义词
    for synonym in synonyms:
        try:
            synonym_ids = tokenizer.encode(synonym, add_special_tokens=False)
            if synonym_ids:
                # 使用第一个token的embedding
                synonym_id = synonym_ids[0]
                if synonym_id < old_vocab_size:
                    print(f"为 {special_token} 使用近义词 '{synonym}' 的embedding")
                    return embeddings[synonym_id].unsqueeze(0)
        except:
            continue

    # 如果所有近义词都不在词汇表中，使用语义最相似的词
    print(f"为 {special_token} 寻找语义相似的词...")

    # 简单的基于字符串相似度的搜索
    special_token_meaning = special_token.strip('[]')
    best_score = -1
    best_embedding = None

    # 搜索前10000个token（避免搜索整个词汇表，太慢）
    search_range = min(10000, old_vocab_size)
    for i in range(search_range):
        token_str = tokenizer.decode([i])
        # 计算简单的字符串相似度（包含关系）
        score = 0
        if special_token_meaning in token_str:
            score = 0.5
        elif any(char in token_str for char in special_token_meaning):
            score = 0.3

        if score > best_score:
            best_score = score
            best_embedding = embeddings[i].unsqueeze(0)

    if best_embedding is not None and best_score > 0:
        print(f"为 {special_token} 使用相似词 (相似度: {best_score:.2f}) 的embedding")
        return best_embedding

    # 最后的手段：使用平均embedding
    print(f"为 {special_token} 使用平均embedding")
    return embeddings[:old_vocab_size].mean(dim=0, keepdim=True)


# 为新token初始化embedding（如果添加了新token）
if new_tokens:
    print("开始初始化新token的embedding...")
    with torch.no_grad():
        for i, special_token in enumerate(new_tokens):
            new_token_id = old_vocab_size + i

            # 找到最佳近义词的embedding
            best_embedding = find_best_synonym_embedding(
                special_token, synonym_mapping, tokenizer,
                input_embeddings.weight.data, old_vocab_size
            )

            # 初始化input embeddings
            input_embeddings.weight.data[new_token_id] = best_embedding.clone().squeeze(0)

    print("新token的embedding初始化完成！")

save_path = "/home/user/for_sync/humanvalue/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese-fulltoken"

# 保存模型和tokenizer
print("正在保存模型和tokenizer...")
os.makedirs(save_path, exist_ok=True)
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print(f"已保存到: {save_path}")

# 验证保存结果
print("\n验证保存结果...")
try:
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_path)
    loaded_model = AutoModel.from_pretrained(save_path)

    print(f"加载的tokenizer词汇表大小: {len(loaded_tokenizer)}")
    print(f"加载的模型embedding大小: {loaded_model.get_input_embeddings().weight.shape[0]}")

    # 测试新token
    for special_token in new_tokens[:3]:  # 测试前3个新token
        tokens = loaded_tokenizer.tokenize(special_token + " test")
        print(f"测试分词结果 '{special_token}': {tokens}")

    print("保存和验证完成！")

except Exception as e:
    print(f"验证过程中出现错误: {e}")