# -*- coding: utf-8 -*-
# @Data: 2021/10/19
# @Author: Ning TANG

import numpy as np


# get train instances
def get_train_instances(group_df, num_items, testList, num_negatives):
    # Choose four for the difference set
    # Here to enter train and test and validate
    user_input, item_input, labels = [], [], []

    # pos:neg = 1:num_negatives sample
    for user_train_id, sub_group_df in group_df:
        # total_items and the difference set of the corresponding testList as a selectable difference set
        can_sample_movies = np.setdiff1d(range(num_items), testList[user_train_id], False)
        per_user, per_item, per_label = sample_per_user(user_train_id, sub_group_df, can_sample_movies, num_negatives)
        user_input.extend(per_user)
        item_input.extend(per_item)
        labels.extend(per_label)

    return user_input, item_input, labels

# get retrain instances
def get_retrain_instances(group_df, num_items, testList, num_negatives, modify_user, cf_set):
    # Choose four for the difference set
    # Here to enter train and test and validate
    user_input, item_input, labels = [], [], []

    # pos:neg = 1:num_negatives sample
    for user_train_id, sub_group_df in group_df:
        # total_items and the difference set of the corresponding testList as a selectable difference set
        can_sample_movies = np.setdiff1d(range(num_items), testList[user_train_id], False)
        # If it is the modified user, remove cf_set
        if user_train_id == modify_user:
            sub_group_df = sub_group_df[~sub_group_df['movie_train_id'].isin(cf_set)]

        per_user, per_item, per_label = sample_per_user(user_train_id, sub_group_df, can_sample_movies, num_negatives)
        user_input.extend(per_user)
        item_input.extend(per_item)
        labels.extend(per_label)

    return user_input, item_input, labels


# Rated + negative sampling: pass in a grouping df, make the difference with the whole movie, and then use numpy to sample
def sample_per_user(user_train_id, sub_group_df, can_sample_movies, num_negatives):

    # The number of movies watched by the current user

    per_items = sub_group_df['movie_train_id'].tolist()
    pos_train_num = len(per_items)
    # print('pos_train_num: ', pos_train_num)
    per_labels = [1]*pos_train_num

    # Find the difference between the total movie and the movie watched by the current user,
    # series, list, numpy are acceptable, and numpy is returned
    diff_movies = np.setdiff1d(can_sample_movies, per_items, False)
    # Randomly select 4*user_actions_num from the difference set as 0 in train
    try:
        neg_train_items = list(np.random.choice(diff_movies, size=num_negatives*pos_train_num, replace=False))
    except:
        neg_train_items = list(diff_movies)

    # Current user's inputs
    per_users = [user_train_id]*(pos_train_num+len(neg_train_items))
    per_items.extend(neg_train_items)
    per_labels.extend([0]*(len(neg_train_items)))

    return per_users, per_items, per_labels


def sample_perturbed_user(user_train_id, sub_group_df, can_sample_movies, num_negatives):
    # The number of movies currently watched by the user

    per_items = sub_group_df['movie_train_id'].tolist()
    pos_train_num = len(per_items)

    # Find the difference between the total movie and the movie watched by the current user,
    # series, list, numpy are acceptable, and numpy is returned
    diff_movies = np.setdiff1d(can_sample_movies, per_items, False)
    # Randomly select 4*user_actions_num from the difference set as 0 in train
    try:
        neg_train_items = list(np.random.choice(diff_movies, size=num_negatives * pos_train_num, replace=False))
    except:
        neg_train_items = list(diff_movies)

    # 当前user的inputs
    per_users = [user_train_id] * (len(neg_train_items))
    per_labels = [0] * (len(neg_train_items))

    return per_users, neg_train_items, per_labels


def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)

    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)

    # MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' % i).get_weights()
        model.get_layer('layer%d' % i).set_weights(mlp_layer_weights)

    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5 * new_weights, 0.5 * new_b])
    return model


if __name__ == '__main__':
    import configs
    from DataSet import Dataset

    dataset = Dataset('Data/%s' % configs.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    # num_users, num_items = train.shape
    user_input, item_input, labels = get_train_instances(train, configs.num_negatives)