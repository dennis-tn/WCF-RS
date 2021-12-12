import numpy as np
import pandas as pd

import configs

from sklearn.linear_model import LogisticRegression, LinearRegression
import scipy.sparse as sp


topK = [5, 10, 20]

top5_top10pos_index, top5_top10pos_weights = [], []
top5_top10neg_index, top5_top10neg_weights = [], []

top10_top10pos_index, top10_top10pos_weights = [], []
top10_top10neg_index, top10_top10neg_weights = [], []

top20_top10pos_index, top20_top10pos_weights = [], []
top20_top10neg_index, top20_top10neg_weights = [], []

top5_pos_list, top5_neg_list = [], []
top10_pos_list, top10_neg_list = [], []
top20_pos_list, top20_neg_list = [], []

# 最小反事实解释列表
top5_min_cf_set_index_s, top5_min_cf_set_weights_s = [], []
top10_min_cf_set_index_s, top10_min_cf_set_weights_s = [], []
top20_min_cf_set_index_s, top20_min_cf_set_weights_s = [], []

top5_gap_list = []
top10_gap_list = []
top20_gap_list = []

top5_has_weights_list = []
top10_has_weights_list = []
top20_has_weights_list = []

num_interaction_list = []

user_item_matrix = sp.load_npz('MedData/user_item_matrix.npz').todok()

for user_id in configs.perturbed_users:
    # input
    perturbed_samples = sp.load_npz(
        'Analysis/Perturbed_Data/perturbed_samples_user_%d_sample_%d.npz' % (user_id, configs.num_samples))

    df_perturb = pd.read_csv('Analysis/CF_Data/perturbed_rank_user_%d_sample_%d.csv' % (user_id, configs.num_samples))

    # number of current user interactions
    num_interaction = user_item_matrix[user_id].size
    num_interaction_list.append(num_interaction)


    for k in topK:
        _shift_rank = 'Top%d_ShiftRank'%(k,)
        # labels
        labels = df_perturb[_shift_rank].values

        # creat Lasso model
        model = LinearRegression()
        model.fit(perturbed_samples, labels)

        # Print model parameters
        print('--- params ---')
        # print(clf.coef_, clf.intercept_)
        weights = model.coef_
        print(weights)

        # Count the number of non-zero weights
        num_coverage = weights[weights != 0].size


        # pos weights
        pos_weights = weights[weights > 0]
        # neg weights
        neg_weights = weights[weights < 0]

        sorted_pos_weights_index = np.argsort(-1 * pos_weights)
        sorted_pos_weights = pos_weights[sorted_pos_weights_index]

        sorted_neg_weights_index = np.argsort(neg_weights)
        sorted_neg_weights = neg_weights[sorted_neg_weights_index]

        # Cumulative positive and negative:
        total_positive = weights[weights > 0].sum()
        total_negative = weights[weights < 0].sum()

        # Weight sum
        sum_neg_pos = total_negative + total_positive


        # Because shift rank reaches 1, the rank will change, so calculate 1-sum_neg_pos
        if num_coverage != 0:
            min_cf_set_index = []
            min_cf_set_weights = []
            gap = 1 - sum_neg_pos
            # The weight is not all 0, it is not interpreted as [], add if there is
            if gap > 0:
                for i in range(len(sorted_neg_weights)):
                    min_cf_set_index.append(sorted_neg_weights_index[i])
                    min_cf_set_weights.append(sorted_neg_weights[i])
                    gap += sorted_neg_weights[i]
                    if gap <= 0:
                        break
        else:
            # The weights are all 0, no explanation can be generated
            min_cf_set_index, min_cf_set_weights = -1, -1
            gap = 0


        print('total_positive: ', total_positive)
        print('total_negative: ', total_negative)

        if k == 5:
            top5_top10pos_index.append(list(sorted_pos_weights_index[:10]))
            top5_top10pos_weights.append(list(sorted_pos_weights[:10]))
            top5_top10neg_index.append(list(sorted_neg_weights_index[:10]))
            top5_top10neg_weights.append(list(sorted_neg_weights[:10]))

            top5_pos_list.append(total_positive)
            top5_neg_list.append(total_negative)
            top5_gap_list.append(gap)
            top5_has_weights_list.append(num_coverage)

            top5_min_cf_set_index_s.append(min_cf_set_index)
            top5_min_cf_set_weights_s.append(min_cf_set_weights)

        elif k == 10:
            top10_top10pos_index.append(list(sorted_pos_weights_index[:10]))
            top10_top10pos_weights.append(list(sorted_pos_weights[:10]))
            top10_top10neg_index.append(list(sorted_neg_weights_index[:10]))
            top10_top10neg_weights.append(list(sorted_neg_weights[:10]))
            top10_pos_list.append(total_positive)
            top10_neg_list.append(total_negative)
            top10_gap_list.append(gap)
            top10_has_weights_list.append(num_coverage)

            top10_min_cf_set_index_s.append(min_cf_set_index)
            top10_min_cf_set_weights_s.append(min_cf_set_weights)

        elif k == 20:
            top20_top10pos_index.append(list(sorted_pos_weights_index[:10]))
            top20_top10pos_weights.append(list(sorted_pos_weights[:10]))
            top20_top10neg_index.append(list(sorted_neg_weights_index[:10]))
            top20_top10neg_weights.append(list(sorted_neg_weights[:10]))
            top20_pos_list.append(total_positive)
            top20_neg_list.append(total_negative)
            top20_gap_list.append(gap)
            top20_has_weights_list.append(num_coverage)

            top20_min_cf_set_index_s.append(min_cf_set_index)
            top20_min_cf_set_weights_s.append(min_cf_set_weights)

df_dict = {'user_id':configs.perturbed_users, 'num_interactions':num_interaction_list,
           'top5_has_weights': top5_has_weights_list, 'top5_minCFset_index': top5_min_cf_set_index_s, 'top5_minCFset_weight': top5_min_cf_set_weights_s,
            'top5_gap': top5_gap_list,
           'top5_top10pos_index':top5_top10pos_index, 'top5_top10pos_weights':top5_top10pos_weights,
           'top5_top10neg_index':top5_top10neg_index, 'top5_top10neg_weights':top5_top10neg_weights,
           'top5_sum_pos':top5_pos_list, 'top5_sum_neg':top5_neg_list,
           'top10_has_weights': top10_has_weights_list, 'top10_minCFset_index': top10_min_cf_set_index_s, 'top10_minCFset_weight': top10_min_cf_set_weights_s,
            'top10_gap': top10_gap_list,
           'top10_top10pos_index':top10_top10pos_index, 'top10_top10pos_weights':top10_top10pos_weights,
           'top10_top10neg_index':top10_top10neg_index, 'top10_top10neg_weights':top10_top10neg_weights,
           'top10_sum_pos': top10_pos_list, 'top10_sum_neg': top10_neg_list,
           'top20_has_weights': top20_has_weights_list, 'top20_minCFset_index': top20_min_cf_set_index_s, 'top20_minCFset_weight': top20_min_cf_set_weights_s,
            'top20_gap': top20_gap_list,
           'top20_top10pos_index':top20_top10pos_index, 'top20_top10pos_weights':top20_top10pos_weights,
           'top20_top10neg_index':top20_top10neg_index, 'top20_top10neg_weights':top20_top10neg_weights,
           'top20_sum_pos': top20_pos_list, 'top20_sum_neg': top20_neg_list
           }

df_weights = pd.DataFrame(df_dict)

df_weights.to_csv('Analysis/cf_weights_sample_%d(0-100).csv' % (configs.num_samples,), index=False)