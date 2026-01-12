import time
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import json
import random
from utils import *
from basic_config import *
from scipy.stats import kurtosis
import warnings
#from GMM import auto_cluster_values
#from auto_cluster import auto_cluster_values

warnings.filterwarnings('ignore')

#plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
#plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# ==================== 大模型解释维度含义 ====================
def llm_explain_prompt(weight_dict):
    # 用于让大模型解释维度的含义
    prompt = f"""以下是一个人的Schwartz价值观分析结果，请结合该价分析结果中各个价值观和对应的价值观权重信息，用不超过15个中文汉字概述这个人的主要价值观：
    价值观组合：{weight_dict}

    只回答你的答案，不要回答其他内容，不要解释你的回答

    参考案例：
    价值观组合1：'自主'：0, '刺激'：1, '享乐主义'：0, '成就'：0,'权力'：0, '安全'：0, '传统'：0, '遵从'：0,'友善'：0, '博爱'：0
    参考回答1：追求刺激为主

    价值观组合2：'自主'：0.5, '刺激'：0.4, '享乐主义'：0.1, '成就'：0,'权力'：0, '安全'：0, '传统'：0, '遵从'：0,'友善'：0, '博爱'：0
    参考回答2：兼顾自主和刺激

    价值观组合3：'自主'：0.2, '刺激'：0.0, '享乐主义'：0.1, '成就'：0.1,'权力'：0.1, '安全'：0.1, '传统'：0.1, '遵从'：0.1,'友善'：0.1, '博爱'：0.1
    参考回答3：略偏自主，整体均衡

    价值观组合3：'自主'：0.1, '刺激'：0.1, '享乐主义'：0.1, '成就'：0.1,'权力'：0.1, '安全'：0.1, '传统'：0.1, '遵从'：0.1,'友善'：0.1, '博爱'：0.1
    参考回答3：均衡价值观追求


    Schwartz价值观参考标准：
    普通性：指为了所有人类和自然的福祉而理解、欣赏、忍耐、保护。例如：社会公正、心胸开阔、世界和平、智慧、美好的世界、与自然和谐一体、保护环境、公平。
    传统：指尊重、赞成和接受文化或宗教的习俗和理念。例如：接受生活的命运安排、奉献、尊重传统、谦卑、节制等。
    顺从：指对行为、喜好和伤害他人或违背社会期望的倾向加以限制。例如：服从、自律、礼貌、给父母和他人带来荣耀。
    安全：指安全、和谐、社会的稳定、关系的稳定和自我稳定。例如：家庭安全、国家安全，社会秩序、清洁、互惠互利等。
    自我导向：指思想和行为的独立──选择、创造、探索。例如：创造性、好奇、自由、独立、选择自己的目标。
    刺激：指生活中的激动人心、新奇的和挑战性。例如：冒险、变化的和刺激的生活。
    享乐主义：指个人的快乐或感官上的满足。例如：愉快、享受生活等。
    仁慈：指维护和提高那些自己熟识的人们的福利。例如：帮助、原谅、忠诚、诚实、真诚的友谊。
    权力：指社会地位与声望、对他人以及资源的控制和统治。例如：社会权力、财富、权威等
    成就：指根据社会的标准，通过实际的竞争所获得的个人成功。例如：成功的、有能力的、 有抱负的、有影响力的等等
    """
    return prompt

def llm_describe_component(weight_dict):
    # 使用大模型来描述维度权重组合的含义
    response = ""
    result = ""
    prompt = llm_explain_prompt(weight_dict)
    for times in range(5):
        try:
            response = api_response(user_prompt=prompt)
            result = str(response)
            assert len(response) >= 3
            print(f"weight_dict:{weight_dict}\nllm解释:{result}")
            break
        except:
            print(f"失败，重新尝试第{times + 1}次，回复为{response}，权重为{weight_dict}")
            time.sleep(3)
    return result

def find_optimal_k(data, max_k):
    # 根据轮廓系数自适应确定最佳聚类数
    optimal_k = 0
    try:
        silhouette_scores = []
        # 从2开始，尝试不同的k值
        for k in range(2, min(max_k + 1, len(data))):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            # 计算轮廓系数
            score = silhouette_score(data, labels)
            silhouette_scores.append(score)
            print(f"k = {k}, Silhouette Score = {score:.4f}")

        # 找到轮廓系数最大的k值
        optimal_k = np.argmax(silhouette_scores) + 2  # +2因为k从2开始
        print(f"\nOptimal k based on Silhouette Score: {optimal_k}")

        # 绘制轮廓系数随k值变化的曲线
        # plt.figure(figsize=(8, 5))
        # plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
        # plt.xlabel('Number of clusters (k)')
        # plt.ylabel('Silhouette Score')
        # plt.title('Silhouette Score for Optimal k')
        # plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
        # plt.legend()
        # plt.show()
    except Exception as e:
        print(f'轮廓系数自适应确定最佳聚类数出错:{e}')

    return optimal_k

# ==================== 数据处理函数 ====================

def trans_hanzi_value(input):
    # 将中文表达的价值观转化为数值，只对基于大模型的价值观预测有用
    # 具体转化数值可以再调整
    if type(input) == float or type(input) == int:

        return input
    try:
        input = float(input)
        return input
    except:
        # 说明本身不是数字
        hanzi_value_dict = {
            "该价值观毫无体现": 0.2,
            "该价值观极微弱痕迹": 0.3,
            "该价值观轻微迹象": 0.4,
            "该价值观明确表达": 0.6,
            "该价值观显著表达": 0.9,
            "该价值观深刻贯彻": 1.0
        }
        return hanzi_value_dict[input]


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

def enhance_event_data(event_data:dict):
    # 提高显示效果，对结果进行缩放，主要功能为放缩价值观数值，另外截断过长的字符串和数值


    event_data["事件"] = event_data["event"]

    # 如果部分价值观为0，会导致雷达图变成刺猬图，反应实际但确实不好看。这部分可以依据实际情况调整


    # 对字符串和数值进行保留截断
    event_name = event_data['事件']
    if 'person' in event_data:
        for i, value_data in enumerate(event_data["person"]["value"]):
            event_data["person"]["value"][i] = [data_i + 0.2 if data_i <= 0.8 else data_i for data_i in value_data]
        title_list = event_data['person']['title']
        # event_data['user']['title'] = [title_i.replace("#" + event_name + "#", '').replace("【】", '') for title_i in title_list]
        for title_index, title in enumerate(title_list):
            title = title.replace(f'#{event_name}#', '')
            # 去除Tag，让信息更密集
            title = remove_event_tags(title)
            # 构造类微博方式
            title = title if len(title) <= 40 else title[:20] + '...' + title[-12:]
            title_list[title_index] = title
        event_data['person']['title'] = title_list
        # 截断数值
        event_data['person']['value'] = [[round(value_i, 3) for value_i in value_i_one]  for value_i_one in event_data['person']['value']]
        event_data['person']['sharp'] = [round(value_i, 3) for value_i in event_data['person']['sharp']]
        event_data['person'] = drop_duplicate_value(event_data['person'])

    if 'human_pie' in event_data:
        for index_pie,  event_pie in enumerate(event_data['human_pie']):
            for sub_index, sub_data_part in enumerate(event_pie['subData']):
                # 截断数值
                event_data['human_pie'][index_pie]['subData'][sub_index]["value"] = round(sub_data_part['value'], 3)

    return event_data


# ==================== 价值观异常度计算函数 ====================

def compute_normalized_kurtosis(vector):
    #计算归一化峰度（内部尖锐度）
    #输入vector: 一维数组，雷达图的一条数据
    #返回归一化峰度值，范围[0,1]，越大表示越尖锐


    # 数据不够
    if len(vector) < 4:
        return 0.0

    k = kurtosis(list(vector), fisher=True, bias=False)
    normalized_k = 1 / (1 + np.exp(-k / 5))
    return normalized_k


def compute_mahalanobis_anomaly(vector, data_matrix):
    # 计算马氏距离异常度（考虑维度相关性）
    # vector: 一维数组，当前数据，data_matrix: 整个数据集矩阵，用于计算统计量
    # 输出为归一化的马氏距离异常度，范围[0,1]

    n_samples, n_dims = data_matrix.shape

    if n_samples < n_dims + 1:
        # 样本太少，退化为欧氏距离
        mean_vector = np.mean(data_matrix, axis=0)
        std_vector = np.std(data_matrix, axis=0) + 1e-10
        z_scores = (vector - mean_vector) / std_vector
        euclidean_anomaly = np.sqrt(np.mean(z_scores ** 2))
        normalized = 1 / (1 + np.exp(-euclidean_anomaly / 3))
        return normalized

    # 计算协方差矩阵
    cov_matrix = np.cov(data_matrix, rowvar=False, bias=True)

    # 添加正则化
    cov_matrix += np.eye(n_dims) * 1e-6

    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except:
        cov_inv = np.linalg.pinv(cov_matrix)

    # 计算均值
    mean_vector = np.mean(data_matrix, axis=0)

    # 计算马氏距离
    diff = vector - mean_vector
    mahal_dist_sq = np.dot(np.dot(diff, cov_inv), diff)
    mahal_dist = np.sqrt(np.abs(mahal_dist_sq))

    # 归一化：使用卡方分布的百分位数
    # 马氏距离平方近似服从自由度为n_dims的卡方分布
    # 使用卡方分布的99%分位数作为参考
    chi2_99 = 2 * (n_dims ** 1.5)  # 近似值
    normalized = min(mahal_dist / np.sqrt(chi2_99), 1.0)

    # 使用sigmoid进一步压缩
    normalized = 1 / (1 + np.exp(-normalized * 5))

    return normalized

def compute_comprehensive_sharpness(vector, data_matrix, internal_weight=0.5):

    #计算综合尖锐度

    #参数：vector: 一维数组，当前数据; data_matrix: 整个数据集矩阵;
    #internal_weight: 内部尖锐度权重（0-1）; external_method: 外部尖锐度计算方法

    #返回：字典包含各项分数
    # 计算内部尖锐度（归一化峰度）
    internal_score = compute_normalized_kurtosis(vector)

    # 计算外部尖锐度（马氏距离）
    external_score = compute_mahalanobis_anomaly(
        vector, data_matrix
    )

    # 计算综合分数
    external_weight = 1 - internal_weight
    comprehensive_score = (internal_weight * internal_score +
                           external_weight * external_score)

    return {
        'internal_sharpness': internal_score,
        'external_sharpness': external_score,
        'comprehensive_sharpness': comprehensive_score,
    }

def analyze_radar_sharpness(df, label_columns, internal_weight=0.5):

    #批量分析雷达图数据尖锐度

    #参数：df: 包含数据的DataFrame；label_columns: 雷达图维度列名列表
    #internal_weight: 内部尖锐度权重；external_method: 外部尖锐度计算方法

    #返回：添加了尖锐度评分的DataFrame，按综合尖锐度排序

    # 提取数据矩阵
    data_matrix = df[label_columns].values
    n_samples = len(df)

    print(f"开始分析 {n_samples} 条数据...")
    print(f"维度数量: {len(label_columns)}")
    print(f"内部权重: {internal_weight}, 外部权重: {1-internal_weight}")

    df["internal_sharpness"] = 0
    df["external_sharpness"] = 0
    df["comprehensive_sharpness"] = 0

    # 逐行计算
    for idx, row in df.iterrows():
        vector = row[label_columns].values

        # 计算综合尖锐度
        scores = compute_comprehensive_sharpness(
            vector,
            data_matrix,
            internal_weight=internal_weight,
        )

        # 记录结果行
        df.loc[idx, 'internal_sharpness'] = scores['internal_sharpness']
        df.loc[idx, 'external_sharpness'] = scores['external_sharpness']
        df.loc[idx, 'comprehensive_sharpness'] = scores['comprehensive_sharpness']

    # 创建结果DataFrame
    results_df = df

    # 按综合尖锐度排序
    sorted_df = results_df.sort_values('comprehensive_sharpness', ascending=False)

    # 添加排名
    sorted_df['sharpness_rank'] = range(1, len(sorted_df) + 1)

    # 重置索引
    sorted_df = sorted_df.reset_index(drop=True)

    # 打印摘要
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print(f"综合尖锐度统计:")
    print(
        f"  范围: {sorted_df['comprehensive_sharpness'].min():.4f} - {sorted_df['comprehensive_sharpness'].max():.4f}")
    print(f"  均值: {sorted_df['comprehensive_sharpness'].mean():.4f}")
    print(f"  标准差: {sorted_df['comprehensive_sharpness'].std():.4f}")



    print(f"\n最尖锐的3个样本:")
    for i in range(min(3, len(sorted_df))):
        score = sorted_df.loc[i, 'comprehensive_sharpness']
        print(f"  排名{i + 1}: 综合={score:.4f}")

    print("=" * 60)

    return sorted_df








# ==================== 主要的聚类和分析逻辑 ====================




def get_dir_group_value(data_dir='结果文件'):
    # 对文件夹进行群体价值观分析，主要功能为使用聚类算法对价值观进行聚类并汇总分析结果形成事件级结果
    data_file = get_files_in_folder(data_dir)
    data_total = {}

    for idx, data_path in enumerate(data_file):
        # 跳过已处理的和未经过上一步处理的
        if 'ensemble' not in data_path:
            continue
        if '_cluster' in data_path:
            continue

        if True:
            # 对单个文件进行群体分析
            result_list = get_jiazhiguan_file_multi(data_path)
            for result in result_list:
                # 读取事件名
                event = result['event']
                # 记录该事件的结果
                data_total[event] = result

    # 导出结果
    with open(data_dir + f'{os.sep}result_value.json','w', encoding='utf-8') as f:
        json.dump(data_total, f, ensure_ascii=False, indent=4)
    return data_dir + f'{os.sep}result_value.json', data_total


def get_jiazhiguan_file_multi(file_path, total_draw_num=20, filter_dict=dict()):
    # 聚类和分析单个文件的价值观，主要功能为使用聚类算法，对单个文件分析价值观组成，并汇总
    # 输入为文件地址，画图个数和筛选条件
    # 输出为存储有结果dict的list
    print(f"process:{file_path}")
    if os.path.exists(file_path.replace(".csv", "_cluster.csv")):
        # 默认重新处理，这部分耗时不久
        pass
        # return load_jiazhiguan_file(file_path.replace(".csv", "_cluster.csv"), total_draw_num)

    # 确定事件名字
    event = file_path.split(f'{os.sep}')[-1].split('_')[0]
    # 读取文件
    try:
        file_path = file_path.replace(".xlsx", '.csv')
        try:
            df = pd.read_csv(file_path, encoding='utf_8_sig')
        except:
            df = pd.read_csv(file_path, encoding='gbk')
    except:
        file_path = file_path.replace(".csv", '.xlsx')
        df = pd.read_excel(file_path)



    df = df.fillna("")
    df = df.drop_duplicates(subset=['id'])
    df = df.reset_index(drop=True)

    # 设置新的属性列
    df['value'] = ""

    if llm_or_bert:
        LABEL_COLUMNS = ['自主', '刺激', '享乐主义', '成就', '权力', '安全', '传统', '遵从', '友善', '博爱']
    else:
        LABEL_COLUMNS = ['思想自主', '行动自主', '刺激', '享乐主义', '成就', '支配权力', '资源权力', '面子', '个人安全',
                         '社会安全', '传统', '规则遵从', '人际遵从', '谦逊', '友善-关怀', '友善-可依赖', '博爱-关注',
                         '博爱-大自然', '博爱-宽容', '博爱-客观性']


    # 取出各事件的数据
    # 目前默认一个数据文件里只有一个事件，如果不止的话，请修改one_file_with_multi_event，避免不同事件结果混在一起
    if "target" in df.columns:
        # 处理valuetalk的结果
        df["话题"] = df["target"]
    event_data_to_analyiz = []
    if one_file_with_multi_event:
        # 一个文件里有多个事件，需要拆开来分别聚类
        for event in df["话题"].unique().tolist():
            df_if = df['话题'].apply(lambda x: event in x)
            event_data = df[df_if]
            event_data = event_data.reset_index(drop=True)
            event_data_to_analyiz.append(event_data)
    else:
        # 一个文件只有一个事件，直接聚类
        event_data_to_analyiz = [df]

    all_event_data_list = []
    all_result_list = []

    """
    
    """
    for event_data in event_data_to_analyiz:
        # 假设 input_data_1 是一个10或20维的Pandas DataFrame
        # input_data_1 = pd.DataFrame(np.random.rand(100, 10))  # 示例数据
        if len(event_data) < 6:
            continue

        if llm_or_bert:
            # 取出大模型价值观不为空的
            event_data = event_data[event_data['大模型汇总价值观'] != '']
            # 将价值观数据取出来
            input_data_1 = pd.DataFrame(event_data['大模型汇总价值观'].apply(lambda x: eval(x)))
            # 逐个按顺序提取出来，并从汉字转化为数值
            for label in LABEL_COLUMNS:
                input_data_1[label] = event_data['大模型汇总价值观'].apply(lambda x: trans_hanzi_value(eval(x)[label]))
                event_data[label] = event_data['大模型汇总价值观'].apply(lambda x: trans_hanzi_value(eval(x)[label]))
        else:
            # 如果只想看中间的bert预测结果
            input_data_1 = pd.DataFrame(event_data['凝练后推文_value'].apply(lambda x: np.array(ast.literal_eval(x))))
            for i, label in enumerate(LABEL_COLUMNS):
                input_data_1[label] = event_data['凝练后推文_value'].apply(lambda x: np.array(ast.literal_eval(x)[i]))
        # 确保排序正确
        input_data_1 = input_data_1[LABEL_COLUMNS]

        # 设置最大聚类数
        max_k = 7

        if False:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(len(input_data_1) / 2, 30))
            combined_data = input_data_1
            input_data_1 = tsne.fit_transform(combined_data)
            input_data_1 = pd.DataFrame(input_data_1)
        if False:
            pca = PCA(n_components=4)
            input_data_1 = pca.fit_transform(input_data_1)
            input_data_1 = pd.DataFrame(input_data_1)


        # 根据肘部法则或轮廓系数选择合适的k值
        optimal_k = find_optimal_k(input_data_1, max_k)
        optimal_k = max(optimal_k, 4)

        # 使用最佳k值进行K-means聚类
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans.fit(input_data_1)

        # 输出聚类中心
        cluster_centers = kmeans.cluster_centers_
        print("Cluster Centers:\n", cluster_centers)

        # 输出每个数据所属的类别
        labels = kmeans.labels_
        input_data_1['Cluster'] = labels
        print("\nData with Cluster Labels:\n", input_data_1.head())
        # 记录聚类结果

        input_data_1['Cluster'] = labels
        event_data['cluster_label'] = labels
        # 由于系统显示功能调整，以下部分暂时用不上了
        if False:
            preson_value_data = []
            pca_axis = ["", ""]
            pca_centers = []
            pca_data = []

            tsne_axis = ["", ""]
            tsne_data = []
            tsne_centers = []

            labels = []
            # 使用PCA将数据和聚类中心降维到二维
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(input_data_1.iloc[:, :-1])  # 降维数据（排除最后一列的聚类标签）
            pca_centers = pca.transform(cluster_centers)  # 降维聚类中心

            for i, component in enumerate(pca.components_):
                print(f"PC{i + 1}:")
                for j, weight in enumerate(component):
                    print(f"  Feature {j}: {weight:.4f}")
                print()

            # 考虑第一主成分和第二主成分
            component_1 = pca.components_[0]
            weight_dict_1 = {LABEL_COLUMNS[i]: component_1[i] for i in range(len(LABEL_COLUMNS))}

            component_2 = pca.components_[1]
            weight_dict_2 = {LABEL_COLUMNS[i]: component_2[i] for i in range(len(LABEL_COLUMNS))}

            # 为其生成文字描述
            xlabel = llm_describe_component(weight_dict_1)
            ylabel = llm_describe_component(weight_dict_2)

            # 确定其维度
            pca_axis = [xlabel, ylabel]

            # 使用t-SNE将数据和聚类中心降维到二维
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(len(input_data_1) / 2, 30))

            # 由于t-SNE不支持直接转换新数据（如聚类中心），我们需要将聚类中心与原始数据一起降维
            # 将聚类中心添加到原始数据中，然后进行t-SNE降维
            combined_data = np.vstack([input_data_1.iloc[:, :-1], cluster_centers])
            tsne_combined = tsne.fit_transform(combined_data)

            # 分离降维后的数据和聚类中心
            tsne_data = tsne_combined[:-optimal_k]  # 数据点
            tsne_centers = tsne_combined[-optimal_k:]  # 聚类中心

            # 将t-SNE降维结果分为两个目标变量
            tsne_1 = tsne_data[:, 0]  # 第一维
            tsne_2 = tsne_data[:, 1]  # 第二维

            # 训练LightGBM模型，预测t-SNE的第一维
            dtrain_tsne_1 = lgb.Dataset(input_data_1[LABEL_COLUMNS].values, label=tsne_1)
            params = {
                'objective': 'regression',  # 回归任务
                'metric': 'rmse',  # 使用均方根误差作为评估指标
                'boosting_type': 'gbdt',  # 使用GBDT算法
                'min_data_in_leaf': 1,  # 允许单样本分裂
                'num_leaves': max(3, len(input_data_1) // 10),  # 树的复杂度
                # 'learning_rate': 0.05,  # 学习率
                'feature_fraction': 0.9,  # 特征采样比例
                'verbose': 1  # 不输出训练日志
            }
            model_tsne_1 = lgb.train(params, dtrain_tsne_1, num_boost_round=100)

            # 训练LightGBM模型，预测t-SNE的第二维
            dtrain_tsne_2 = lgb.Dataset(input_data_1[LABEL_COLUMNS].values, label=tsne_2)
            model_tsne_2 = lgb.train(params, dtrain_tsne_2, num_boost_round=100)

            # 获取特征重要性
            feature_importances_tsne_1 = model_tsne_1.feature_importance(importance_type='gain')
            feature_importances_tsne_1 = feature_importances_tsne_1 / feature_importances_tsne_1.sum()
            feature_importances_tsne_2 = model_tsne_2.feature_importance(importance_type='gain')
            feature_importances_tsne_2 = feature_importances_tsne_2 / feature_importances_tsne_2.sum()

            # 将特征重要性作为维度组成
            component_1 = feature_importances_tsne_1.tolist()
            weight_dict_1 = {LABEL_COLUMNS[i]: component_1[i] for i in range(len(LABEL_COLUMNS))}

            component_2 = feature_importances_tsne_2.tolist()
            weight_dict_2 = {LABEL_COLUMNS[i]: component_2[i] for i in range(len(LABEL_COLUMNS))}

            # 使用大模型解释维度的构成
            xlabel = llm_describe_component(weight_dict_1)
            ylabel = llm_describe_component(weight_dict_2)

            tsne_axis = [xlabel, ylabel]

            event_data['pca_data'] = list(pca_data)
            event_data['pca_data_str'] = df['pca_data'].apply(lambda x: str(x.tolist()))
            event_data['tsne_data'] = list(tsne_data)
            event_data['tsne_data_str'] = df['tsne_data'].apply(lambda x: str(x.tolist()))

            event_data['pca_axis_x'] = pca_axis[0]
            event_data['pca_axis_y'] = pca_axis[1]
            event_data['tsne_axis_x'] = tsne_axis[0]
            event_data['tsne_axis_y'] = tsne_axis[1]

            event_data['pca_center'] = str(pca_centers.tolist())
            event_data['tsne_center'] = str(tsne_centers.tolist())




        # 该功能由于系统显示功能调整，暂时用不上了，这是原本用于展示典型帖子用的，现在已调整为展示异常帖子
        def get_top_samples_per_cluster(input_data, cluster_labels, cluster_centers, df_copy, n_samples=10):
            # 获取每个聚类中心最相近的n_samples个原始样本和序号

            # 计算所有样本到所有聚类中心的距离
            distances = pairwise_distances(input_data, cluster_centers, metric='cosine')

            # 存储结果
            cluster_top_samples = {}

            for cluster_id in range(len(cluster_centers)):
                # 获取当前类别的所有样本索引
                cluster_mask = (cluster_labels == cluster_id)
                cluster_indices = np.where(cluster_mask)[0]

                # 如果当前类样本数不足n_samples，则全部取出
                n = min(int(n_samples/len(cluster_centers)), len(cluster_indices))

                # 获取当前类样本到本类中心的距离
                cluster_distances = distances[cluster_mask, cluster_id]

                # 按距离升序排序，取前n个最近的样本
                sorted_indices = np.argsort(cluster_distances)[:n]
                selected_indices = cluster_indices[sorted_indices]

                # 存储结果
                cluster_top_samples[cluster_id] = {
                    'indices': selected_indices.tolist(),
                    'user_id': df_copy.loc[selected_indices.tolist(), 'user_id'].apply(lambda x:str(x)).tolist(),
                    'samples': input_data.iloc[selected_indices].values.tolist(),
                    'distances': cluster_distances[sorted_indices].tolist(),
                    'contents':df_copy.loc[selected_indices.tolist(), 'content'].tolist(),
                    'id':df_copy.loc[selected_indices.tolist(), 'id'].tolist()
                }

            return cluster_top_samples

        """
        cluster_labels = input_data_1.loc[:, 'Cluster'].values  # 取出聚类标签
        column_without_cluster = list(input_data_1.columns).remove('Cluster')
        # 获取每个聚类中心最近的20个样本
        top_samples = get_top_samples_per_cluster(
            input_data=input_data_1.iloc[:, column_without_cluster],  # 原始数据（排除标签列）
            cluster_labels=cluster_labels,
            cluster_centers=cluster_centers,
            n_samples=20,
            df_copy=df
        )
        """

        """
        # 绘制散点图
        plt.figure(figsize=(10, 6))
    
        for cluster_id, info in top_samples.items():
            top_indices = info['indices']
            plt.scatter(
                tsne_data[top_indices, 0],
                tsne_data[top_indices, 1],
                edgecolors='black',
                linewidths=1,
                facecolors='none',
                s=100,
                label=f'Cluster {cluster_id} Top Samples'
            )
        
    
        for cluster in range(optimal_k):
            # 绘制每个类别的数据点
            plt.scatter(tsne_data[labels == cluster, 0], tsne_data[labels == cluster, 1],
                        label=f'Cluster {cluster + 1}', alpha=0.7)
            # 标注聚类中心
            plt.scatter(tsne_centers[cluster, 0], tsne_centers[cluster, 1], marker='X', s=200, c='black', label="")
    
        plt.title(event + "——聚类结果")
        """
        event_data = event_data.reset_index(drop=True)
        # 构造类weibo的url
        event_data['weibo_url'] = event_data.apply(lambda x: f"https://weibo.com/{x['user_id']}/{x['bid']}", axis=1)
        all_event_data_list.append(event_data)

        # ***********以下为构造提交给服务器的数据***********
        if False:
            # 数据太多了，需要对其进行采样
            event_data['keys'] = range(len(event_data))
            # 对各类随机采样
            choice_key = []
            label_count = optimal_k
            cluster_num = [len(event_data[event_data['cluster_label'] == l]) for l in range(label_count)]
            # 确定各类采样个数
            cluster_num = [int(d / sum(cluster_num) * total_draw_num) for d in cluster_num]
            # 对各类的数据进行随机采样
            for l in range(label_count):
                choice_key += random.sample(event_data[event_data['cluster_label'] == l]['keys'].tolist(),
                                            min(max(1, cluster_num[l]), len(event_data[event_data['cluster_label'] == l]['keys'])))

            # 只保留采样的部分
            event_data = devent_data.iloc[choice_key]
            event_data = event_data.reset_index(drop=True)

        # 分析个体结果
        person_value_data = event_data[['content', 'weibo_url', '用户昵称', '发布时间', 'id', 'user_id'] + LABEL_COLUMNS]
        person_value_data = person_value_data.drop_duplicates(subset=['content'])
        # 分析价值观异常程度
        person_value_data = analyze_radar_sharpness(person_value_data, LABEL_COLUMNS, sharp_internel_weight)
        # 取前20条最异常的价值观结果
        person_value_data = person_value_data.loc[:min(20, len(person_value_data))]
        person_value_data = person_value_data.reset_index()

        # 记录异常个体的结果
        for i_person in range(len(person_value_data)):
            person_name = person_value_data.loc[i_person, '用户昵称']
            # 对名字进行截断
            if len(person_name) >= 5:
                person_name = person_name[:4] + '..'
            # 构造发帖形式
            person_value_data.loc[i_person, 'content'] = person_name + '：' + person_value_data.loc[i_person, 'content']
        # 构造发帖列表
        person_dict = {
            'title': person_value_data['content'].values.tolist(),
            'value': person_value_data[LABEL_COLUMNS].values.tolist(),
            'time': person_value_data['发布时间'].values.tolist(),
            'url': person_value_data['weibo_url'].values.tolist(),
            'id': person_value_data['id'].values.tolist(),
            'user_id': person_value_data['user_id'].values.tolist(),
            'sharp': person_value_data['comprehensive_sharpness'].values.tolist(),
        }

        def get_human_pie(df_t, top_n=5):
            # 统计人群分布饼图

            # 统计占比最多的top_n类人群数量
            human_count = df_t['cluster_label'].value_counts().nlargest(top_n).to_dict()
            human_pie = []

            # 将价值观转化为二分类，来统计持有价值观情况
            for label in LABEL_COLUMNS:
                df_t[label] = df_t[label].apply(lambda x: 0 if x <= 0.5 else 1)

            # 对每类人群统计分布
            for human_i, human_type in enumerate(list(human_count.keys())):

                # 取出该类人群
                df_c = df_t[df_t['cluster_label'] == human_type]
                if 'level_0' in df_c.columns:
                    df_c = df_c.drop(columns=['level_0'])
                df_c = df_c.reset_index()

                # 计算各个价值观的占比
                value_count = np.array([df_c[value_col].sum() / len(df_c) for value_col in LABEL_COLUMNS])
                # 按照从大到小排序
                value_sort_index = np.argsort(-1 * value_count)[:top_n]
                # 记录该类人群的价值观分布情况
                human_pie.append({
                    "value": human_count[human_type],  # 总数
                    "name": f"人群{human_i + 1}",
                    'subData': [{"value": value_count[human_value_i], 'name': LABEL_COLUMNS[human_value_i]} for
                                human_value_i in value_sort_index]
                })

            return human_pie

        # 汇总结果
        result_dict = {
            'event': event,
            'person': person_dict,
            # 'group': group_dict,
            # 'typical': {'value': tsne_centers.tolist()},
            # 'axis': [str(event_data.iloc[0]['tsne_axis_x']), str(event_data.iloc[0]['tsne_axis_y'])],
            # 'typical_user': top_samples,
            'human_pie': get_human_pie(event_data),
            # 'value_pie': get_value_pie(event_data),
            # 'group_average': input_data_1[LABEL_COLUMNS].mean().tolist()
        }

        # 是否提高显示效果，还是说需要原始数据
        if enhance_performence:
            # 提高显示效果
            result_dict = enhance_event_data(result_dict)

        # 进一步转化为给服务器的数据格式
        # 用于适配服务器格式
        result_dict = trans_data_screen(result_dict)
        all_result_list.append(result_dict)





    # 合并各个事件的结果
    df = pd.concat(all_event_data_list)
    # 保存结果
    df.to_csv(file_path.replace('.csv', '_cluster.csv'), encoding='utf_8_sig')





    return all_result_list

if __name__ == '__main__':
    #get_dir_group_value("D:\Pycharm\watch_system\human_value_predict\human_value_data\cluster_events_test_学生")
    get_jiazhiguan_file_multi(r"D:\Pycharm\watch_system\human_value_predict\human_value_data\valuetalk_test_only\valuetalk_test_only_v10_originalensemble.csv")
    get_jiazhiguan_file_multi(r"D:\Pycharm\watch_system\human_value_predict\human_value_data\valuetalk_test_only\valuetalk_test_only_v10_ensemble.csv")
