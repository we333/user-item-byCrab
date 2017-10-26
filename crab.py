#!/usr/bin/env python
#coding=utf-8   

import sys
import numpy as np

class Bunch(dict):
    """
    Container object for datasets: dictionary-like object
    that exposes its keys and attributes. """

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

# 沿用crab的函数,读取data_file,将数据转换为Matrix
def load_test_data(data_file):
    #Read data
    data_m = np.loadtxt(data_file,
                delimiter=',', dtype=str)
    item_ids = []
    user_order_nums = []
    data_songs = {}
    for user_order_num, item_id, rating in data_m:
        if user_order_num not in user_order_nums:
            user_order_nums.append(user_order_num)
        if item_id not in item_ids:
            item_ids.append(item_id)
        u_ix = user_order_nums.index(user_order_num) + 1
        i_ix = item_ids.index(item_id) + 1
        data_songs.setdefault(u_ix, {})
        data_songs[u_ix][i_ix] = float(rating)

    data_t = []
    for no, item_id in enumerate(item_ids):
        data_t.append((no + 1, item_id))
    data_titles = dict(data_t)

    data_u = []
    for no, user_order_num in enumerate(user_order_nums):
        data_u.append((no + 1, user_order_num))
    data_users = dict(data_u)

    fdescr = open('./test_data.rst')

    return Bunch(data=data_songs, item_ids=data_titles,
                 user_order_nums=data_users, DESCR=fdescr.read())

# 读取input_file文件中的[用户名x]
def read_user_name(input_file):
    f_r = open(input_file)
#    for line in f_r:
#        name = line.split('\0')
#        return name
    return 'A'

'''
    通过读取data_file,判断user_name在csv文件里[出现的次序n]
    因为crab接收的输入参数是[用户次序n],不能直接处理user_name
'''
def userName_to_userOrderNum(data_file, user_name):
    recommend_id = {}
    curr_order = 1
    f_r = open(data_file, 'r')
    for line in f_r:
        user = line.split(',')[0]
        if user not in recommend_id:
            recommend_id[user] = curr_order
            curr_order += 1

    for k, v in recommend_id.items():
        if k in user_name:
            return recommend_id[k]

'''
    product_order_num是通过crab推荐,得到的一个推荐商品在csv文件里[出现的次序n]
    因为crab返回的推荐结果是[商品次序n],不能直接返回商品名
    函数功能是:
        1.遍历data_file后记录dict[商品名x] = 商品出现次序n
        2.再遍历dict,找到[商品出现次序n]所对应的商品名x
        3.返回x (真正被推荐的商品的商品名)
'''
def productOrderNum_to_productName(data_file, product_order_num):
    product_id = {}
    curr_order = 1
    f_r = open(data_file, 'r')
    for line in f_r:
        product_name = line.split(',')[1]
        if product_name not in product_id:
            product_id[product_name] = curr_order;
            curr_order += 1

    for k,v in product_id.items():
        if v == product_order_num:
            return k

def user_base(input_file, output_file, data_file):
    # 基础数据-测试数据
    from scikits.crab import datasets
    #	movies = datasets.load_sample_movies()
    user_name = read_user_name(input_file)

    shopping_history = load_test_data(data_file)
    
    user_order_num = userName_to_userOrderNum(data_file, user_name)

    #Build the model
    from scikits.crab.models import MatrixPreferenceDataModel
    model = MatrixPreferenceDataModel(shopping_history.data)

    #Build the similarity
    # 选用算法 pearson_correlation
    from scikits.crab.metrics import pearson_correlation
    from scikits.crab.similarities import UserSimilarity
    similarity = UserSimilarity(model, pearson_correlation)

    # 选择 基于User的推荐
    from scikits.crab.recommenders.knn import UserBasedRecommender
    recommender = UserBasedRecommender(model, similarity, with_preference=True)
    ret = recommender.recommend(user_order_num)	# 输出个结果看看效果 Recommend items for the user 5 (Toby)
    print ret

    if ret: # 因为数据量过少,只有在[得到推荐结果]时才能写入output_file
        product_order_num = ret[0][0]
        product_score = ret[0][1]

        product_name = productOrderNum_to_productName(data_file, product_order_num)

        f_w = open(output_file, 'w')
        f_w.write(str(product_name) + ',' + str(product_score) + '\n')

	# 选择 基于Item 的推荐(同样的基础数据，选择角度不同)
#	from scikits.crab.recommenders.knn import ItemBasedRecommender
#	recommender = ItemBasedRecommender(model, similarity, with_preference=True)
#	print recommender.recommend(1)	# 输出个结果看看效果 Recommend items for the user 5 (Toby)

'''
    [1]:question.csv 读取需要被推荐的用户的name
    [2]:answer.csv 保存推荐结果的文件
    [3]:data.csv 网站所有用户的购买履历
'''
user_base(sys.argv[1], sys.argv[2], sys.argv[3])
#itembase_demo()