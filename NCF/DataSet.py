# -*- coding: utf-8 -*-
# @Data: 2021/10/19
# @Author: Ning TANG

import scipy.sparse as sp
import numpy as np
import pandas as pd

def neg_plus_recent(x, y):
    x = eval(x)
    x.append(y)
    return x


def load_negative_file(filename):
    df_neg_and_recent = pd.read_csv(filename)

    testList = df_neg_and_recent.apply(lambda row: neg_plus_recent(row['negative_movies_100'], row['recent_movie_train_id']), axis=1)
    testList = testList.to_list()

    valList = df_neg_and_recent['recent_movie_train_id'].to_list()

    print('Finish loading 100 negative samples and recent rated movies, length is: ', len(testList))
    print('Finish loading recent rated movies, length is: ', len(valList))

    return testList, valList


def load_rating_file_as_matrix(filename, num_users, num_items):
    """
    Read .rating file and Return dok matrix.
    The first line of .rating file is: num_users\t num_items
    """
    # Get number of users and items
    df_train = pd.read_csv(filename)
    # The number in users is the same as after deduplication of ratings
    df_group_train = df_train.sort_values(by=['user_train_id', 'timestamp'], ascending=[True, False])
    df_group_train = df_group_train[['user_train_id', 'movie_train_id', 'timestamp']]
    df_group_train = df_group_train.groupby(['user_train_id'])

    # Construct matrix
    # dok_matrix inherits from dict, it uses a dictionary to store elements that are not 0 in the matrix:
    # the key of the dictionary is a tuple that stores the information of the element (row, column),
    # and its corresponding value is located in (row, column) in the matrix Element value.
    # the sparse matrix in the dictionary format is very suitable for the addition,
    # deletion and access operations of a single element.
    # It is usually used to gradually add non-zero elements and then convert them to other formats
    # that support fast calculations.

    # user-item co-occurrence matrix mat
    user_item_matrix = sp.dok_matrix((num_users, num_items), dtype=np.int32)
    for i, row in df_train.iterrows():
        user_item_matrix[row['user_train_id'], row['movie_train_id']] = 1

    # save mat
    sp.save_npz('MedData/user_item_matrix.npz', user_item_matrix.tocoo())
    # user_item_matrix = sp.load_npz('MedData/user_item_matrix.npz')
    return user_item_matrix, df_group_train

def load_users(filename):
    df_users = pd.read_csv(filename)
    num_users = len(df_users['train_id'])
    users = df_users['train_id'].tolist()
    return users, num_users

def load_items(filename):
    df_items = pd.read_csv(filename)
    num_items = len(df_items['train_id'])
    items = df_items['train_id'].tolist()
    return items, num_items


class Dataset(object):

    def __init__(self, path):
        """
        Constructor
        """
        self.users, self.num_users = load_users(path + "_users_train.csv")
        self.items, self.num_items = load_items(path + "_movies_train.csv")
        self.trainMatrix, self.df_group_train = load_rating_file_as_matrix(path + "_train_ratings.csv",self.num_users, self.num_items)
        self.testList, self.valList = load_negative_file(path + "_test_negative.csv")
        assert len(self.testList) == len(self.valList)

