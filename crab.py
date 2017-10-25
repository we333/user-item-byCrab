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

def load_test_data(data_file):
    #Read data
    data_m = np.loadtxt(data_file,
                delimiter=';', dtype=str)
    item_ids = []
    user_ids = []
    data_songs = {}
    for user_id, item_id, rating in data_m:
        if user_id not in user_ids:
            user_ids.append(user_id)
        if item_id not in item_ids:
            item_ids.append(item_id)
        u_ix = user_ids.index(user_id) + 1
        i_ix = item_ids.index(item_id) + 1
        data_songs.setdefault(u_ix, {})
        data_songs[u_ix][i_ix] = float(rating)

    data_t = []
    for no, item_id in enumerate(item_ids):
        data_t.append((no + 1, item_id))
    data_titles = dict(data_t)

    data_u = []
    for no, user_id in enumerate(user_ids):
        data_u.append((no + 1, user_id))
    data_users = dict(data_u)

    fdescr = open('./test_data.rst')

    return Bunch(data=data_songs, item_ids=data_titles,
                 user_ids=data_users, DESCR=fdescr.read())

def user_base(data_file, output_file):
    # 基础数据-测试数据
    from scikits.crab import datasets
    #	movies = datasets.load_sample_movies()
    movies = load_test_data(data_file)
    #	print (movies.data)
    #print movies.user_ids
    #print movies.item_ids

    #Build the model
    from scikits.crab.models import MatrixPreferenceDataModel
    model = MatrixPreferenceDataModel(movies.data)

    #Build the similarity
    # 选用算法 pearson_correlation
    from scikits.crab.metrics import pearson_correlation
    from scikits.crab.similarities import UserSimilarity
    similarity = UserSimilarity(model, pearson_correlation)

    # 选择 基于User的推荐
    from scikits.crab.recommenders.knn import UserBasedRecommender
    recommender = UserBasedRecommender(model, similarity, with_preference=True)
    ret = recommender.recommend(1)	# 输出个结果看看效果 Recommend items for the user 5 (Toby)
    ret_user = ret[0][0]
    ret_score = ret[0][1]

    f_w = open(output_file, 'w')
    f_w.write(str(ret_user) + ',' + str(ret_score) + '\n')

	# 选择 基于Item 的推荐(同样的基础数据，选择角度不同)
#	from scikits.crab.recommenders.knn import ItemBasedRecommender
#	recommender = ItemBasedRecommender(model, similarity, with_preference=True)
#	print recommender.recommend(1)	# 输出个结果看看效果 Recommend items for the user 5 (Toby)

	
user_base(sys.argv[1], sys.argv[2])
#itembase_demo()