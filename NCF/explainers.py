import scipy.sparse as sp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from IPython.display import display
import configs
from utils import get_train_instances, sample_per_user, sample_perturbed_user


def get_topk_sim_ids(user_id, sim_vector, K, seen_ids):
    fisrt_sim_order = np.argsort(sim_vector)[::-1]
    # print('fisrt_sim_order: ',fisrt_sim_order[:10])
    descend_sim_ids = []
    i = 0
    while len(descend_sim_ids) < K:
        if fisrt_sim_order[i] not in seen_ids:
            descend_sim_ids.append(fisrt_sim_order[i])
        i += 1
    print(descend_sim_ids)
    return descend_sim_ids


def find_most_k_similar_users(user_id, user_item_matrix, similarity_user_size):
    # cosine similarity
    similarity_matrix = cosine_similarity(user_item_matrix)

    # similarity_matrix[user_id]
    sim_user_ids = []
    seen_ids = [user_id]
    sim1_ids = get_topk_sim_ids(user_id, similarity_matrix[user_id], similarity_user_size, seen_ids)
    sim_user_ids.extend(sim1_ids)

    return list(set(sim_user_ids))


# CF_RS
def get_perturbed_samples(user_id, num_samples=500, similarity_size=[3, 5]):
    user_item_matrix = sp.load_npz('MedData/user_item_matrix.npz').todok()

    sim_user_ids = find_most_k_similar_users(user_id, user_item_matrix, similarity_size[0])

    # user_id 那行
    user_item_row = user_item_matrix.getrow(user_id).toarray()[0]
    index_not_0 = np.where(user_item_row != 0)[0]

    len_not_0 = index_not_0.size

    num_items = user_item_matrix.shape[1]
    # For each sample, how many items are selected as 1
    select_items_num = list(np.random.randint(1, len_not_0, num_samples))


    perturbed_samples = sp.dok_matrix((num_samples, num_items), dtype=np.int32)

    keep_index_s, keep_lens = [], []
    masked_index_s, masked_lens = [], []

    for i, s in enumerate(select_items_num):
        keep_index = list(np.random.choice(index_not_0, s, replace=False))  # replace=False不可以取相同数字
        masked_index = list(set(keep_index) ^ set(index_not_0))

        keep_index_s.append(keep_index)
        masked_index_s.append(masked_index)
        masked_lens.append(len(masked_index))

        # Make the drawn non-zero feature as 1
        for index in keep_index:
            perturbed_samples[i, index] = 1


    return perturbed_samples, sim_user_ids, select_items_num, keep_index_s, masked_index_s, masked_lens


def generate_retrain_data(user_train_id, keep_index, masked_index, sim_user_ids, df_group_train, num_items, testList):
    """
    生成retrain data
    :param perturbed_sample: one row of dok_matrix
    :param sim_user_ids:
    :return:
    """
    # Read the saved pos/neg file
    # df_user_pos_neg = pd.read_csv('MedData/user_pos_neg.csv')
    # Sampling sub_group_df

    user_item_matrix = sp.load_npz('MedData/user_item_matrix.npz').todok()
    retrain_user_input, retrain_item_input, retrain_labels = [], [], []
    # Current perturbed user vector
    retrain_user_input.extend([user_train_id]*(len(keep_index)+len(masked_index)))
    retrain_item_input.extend(keep_index+masked_index)
    retrain_labels.extend([1]*len(keep_index)+[0]*len(masked_index))

    can_sample_movies = np.setdiff1d(range(num_items), testList[user_train_id], False)
    sub_group_df = df_group_train.get_group(user_train_id)
    cur_user, cur_item, cur_label = sample_perturbed_user(user_train_id, sub_group_df, can_sample_movies, configs.num_negatives)
    retrain_user_input.extend(cur_user)
    retrain_item_input.extend(cur_item)
    retrain_labels.extend(cur_label)

    for uid in sim_user_ids:
        can_sample_movies = np.setdiff1d(range(num_items), testList[uid], False)
        sub_group_df = df_group_train.get_group(uid)
        sim_user, sim_item, sim_label = sample_per_user(uid, sub_group_df, can_sample_movies,  configs.num_negatives)
        retrain_user_input.extend(sim_user)
        retrain_item_input.extend(sim_item)
        retrain_labels.extend(sim_label)

    return retrain_user_input, retrain_item_input, retrain_labels


if __name__ == '__main__':
    df_user_pos_neg = pd.read_csv('MedData/user_pos_neg.csv')
    display(df_user_pos_neg.head())
