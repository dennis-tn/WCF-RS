# -*- coding: utf-8 -*-
# @Data: 2021/10/19
# @Author: Ning TANG

'''
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

'''

import heapq
import numpy as np
import rbo

# Global variables that are shared across processes
_model = None
_validation = None
_test = None
_K = None


# HitRatio
# measures whether the test item is present on the top _K list
def getHitRatio(ranklist, validation_item):
    if validation_item in ranklist:
        return 1
    return 0

# NDCG
# measures the position of the hit by assigning higher scores to hits at top ranks
def getNDCG(ranklist, validation_item):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == validation_item:
            return np.log(2) / np.log(i+2)
    return 0


def eval_one_rating(idx):

    # items: 101 items each user
    test_items_101 = _test[idx]

    validation_item = _validation[idx]

    # length: len(items), each value: u
    users = np.full(len(test_items_101), idx, dtype='int32')
    # get predicted score for each item in items
    pX = np.array([users, test_items_101]).T  # 根据模型的输入为两列，因此转置
    # Get prediction scores
    predictions = _model.predict(x=pX, batch_size=pX.shape[0], verbose=0)
    # dict: {item：score}
    scores = list(map(lambda x: x[0], predictions))
    map_item_score = dict(zip(test_items_101, scores))

    # Evaluate top rank list
    # heapq.nlargest:find top _K largest scores
    # ranklist: type:list() / return top _K largest score items
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, validation_item)
    ndcg = getNDCG(ranklist, validation_item)
    # top_K_items_scores = {item:map_item_score[item] for item in ranklist}
    top_K_items_scores = [(item, map_item_score[item]) for item in ranklist]
    return hr, ndcg, top_K_items_scores, ranklist


def evaluate_model(model, validation, test, K):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _validation
    global _test
    global _K
    _model = model
    _validation = validation
    _test = test
    _K = K

    hits, ndcgs, top_K_items_scores_s, ranks = [], [], [], []
    # Single thread
    for idx in range(len(_validation)):
        (hr, ndcg, top_K_items_scores, rank) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
        top_K_items_scores_s.append(top_K_items_scores)
        ranks.append(rank)
    return hits, ndcgs, top_K_items_scores_s, ranks

def evaluate_instance(model, idx, validation, test, K):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _validation
    global _test
    global _K

    _model = model
    _validation = validation
    _test = test
    _K = K

    (hr, ndcg, top_K_items_scores, ranklist) = eval_one_rating(idx)

    return hr, ndcg, top_K_items_scores, ranklist

def getJaccard(o_rank, p_rank):
    inter = len(set(o_rank)&set(p_rank))
    uni = len(set(o_rank)|set(p_rank))
    return inter/uni

def getRbo(o_rank, p_rank):

    return(rbo.RankingSimilarity(o_rank, p_rank).rbo())
