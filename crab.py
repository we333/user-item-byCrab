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

def item_base(data_file, output_file):
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

def itembase_demo():
    from scikits.crab.models.classes import MatrixPreferenceDataModel
    from scikits.crab.recommenders.knn.classes import ItemBasedRecommender
    from scikits.crab.similarities.basic_similarities import ItemSimilarity
    from scikits.crab.recommenders.knn.item_strategies import ItemsNeighborhoodStrategy
    from scikits.crab.metrics.pairwise import euclidean_distances
    movies = {
			'Marcel Caraciolo': \
				{'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, 'The Night Listener': 3.0}, \
			'Paola Pow': \
				{'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0, 'You, Me and Dupree': 3.5}, \
			'Leopoldo Pires': \
				{'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0, 'Superman Returns': 3.5, 'The Night Listener': 4.0}, 
			'Lorena Abreu': \
				{'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'The Night Listener': 4.5, 'Superman Returns': 4.0, 'You, Me and Dupree': 2.5}, \
			'Steve Gates': \
				{'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0, 'You, Me and Dupree': 2.0}, \
			'Sheldom':\
				{'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5}, \
			'Penny Frewman': \
				{'Snakes on a Plane':4.5,'You, Me and Dupree':1.0, 'Superman Returns':4.0}, 'Maria Gabriela': {}
			}
    model = MatrixPreferenceDataModel(movies)
    items_strategy = ItemsNeighborhoodStrategy()
    similarity = ItemSimilarity(model, euclidean_distances)
    recsys = ItemBasedRecommender(model, similarity, items_strategy)
    
    print recsys.most_similar_items('Lady in the Water')
	#Return the recommendations for the given user.
    print recsys.recommend('Leopoldo Pires')
    #Return the 2 explanations for the given recommendation.
    print recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 2)
	#Return the similar recommends
    print recsys.most_similar_items('Lady in the Water')
	#估算评分
    print recsys.estimate_preference('Leopoldo Pires','Lady in the Water')
	
	
item_base(sys.argv[1], sys.argv[2])
#itembase_demo()