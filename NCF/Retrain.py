from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from collections import namedtuple

from NeuCF import NeuMF

import configs
from DataSet import Dataset
from evaluate import evaluate_instance
from utils import get_retrain_instances, load_pretrain_model

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SparseFeat = namedtuple('SparseFeat', ['name', 'nums', 'mf_dim'])

top_K_items_scores_s, ranks = [], []
cf_users = []
shift_rank_s = []
true_cf_count = 0
predict_cf_count = 0

if __name__ == '__main__':

    init_weights = 'Save/%s_NeuMF_%d_%s_ckpt' % (configs.retrain_init, configs.mf_dim, configs.MLP_layers_units)

    # --------------Loading data--------------
    t1 = time()
    dataset = Dataset('Data/%s' % configs.dataset)
    # Use pandas to do, group, and then take the difference set
    train, validation, test = dataset.trainMatrix, dataset.valList, dataset.testList
    df_group_train = dataset.df_group_train
    users, num_users = dataset.users, dataset.num_users
    items, num_items = dataset.items, dataset.num_items

    feature_columns_dict = {'user': SparseFeat('user_id', num_users, 8),
                            'item': SparseFeat('item_id', num_items, 8)}

    print('Load data done [%.1f s]. # user=%d, #item=%d, #train=%d, #test=%d'
          % (time() - t1, num_users, num_items, train.nnz, len(test)))

    # --------------Build model---------------
    model = NeuMF(feature_columns_dict, configs.MLP_layers_units, configs.reg_layers, configs.reg_mf)

    if configs.learner.lower() == "adagrad":
        model.compile(optimizer=optimizers.Adagrad(learning_rate=configs.learning_rate), loss='binary_crossentropy')
    elif configs.learner.lower() == "rmsprop":
        model.compile(optimizer=optimizers.RMSprop(learning_rate=configs.learning_rate), loss='binary_crossentropy')
    elif configs.learner.lower() == "adam":
        model.compile(optimizer=optimizers.Adam(learning_rate=configs.learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['acc'])
    else:
        model.compile(optimizer=optimizers.SGD(learning_rate=configs.learning_rate), loss='binary_crossentropy')

    # ---------------Init performance----------------
    min_loss = None

    # Save the initial weight of the retrain
    if configs.out > 0:
        model.save_weights(init_weights, overwrite=True)

    df_weights = pd.read_csv('Analysis/cf_weights_sample_300(200-300).csv')
    df_rank_best = pd.read_csv('Analysis/predict_rank_best.csv')

    # Top1 rec of each user
    df_rank_top1 = df_rank_best['top_20'].tolist()
    df_rank_top1 = list(map(eval, df_rank_top1))
    # 每个用户的original top1
    df_rank_top1 = list(map(lambda x: x[0], df_rank_top1))

    # top5's minCFset
    top5_minCFset_index = df_weights['top5_minCFset_index'].tolist()
    top5_minCFset_index = list(map(eval, top5_minCFset_index))

    # Starting user id
    start = configs.perturbed_users[0]


    for user_id in configs.perturbed_users:
        top1 = df_rank_top1[user_id]
        if top5_minCFset_index[user_id - start] != [] and top5_minCFset_index[user_id - start] != -1:
            predict_cf_count += 1

            # Reinitialize the model
            model.load_weights(init_weights)

            # -----------------Training model-------------
            for epoch in range(configs.epochs):
                # Generate training instances
                # pos:neg 1:num_negatives
                user_input, item_input, labels = get_retrain_instances(df_group_train, num_items, test,
                                                                       configs.num_negatives, user_id, top5_minCFset_index[user_id - start])
                print('Get_train_instances [%.1f s]' % (time() - t1))
                t1 = time()
                X = np.array([user_input, item_input]).T  # 根据模型的输入为两列，因此转置
                # --------------Fit model----------------
                hist = model.fit(X,
                                 np.array(labels),
                                 batch_size=configs.batch_size,
                                 epochs=1,
                                 verbose=configs.verbose,
                                 shuffle=True)

                # ----------------Evaluation--------------
                # The smaller the cross entropy, the better
                if epoch % configs.verbose == 0:
                    loss = hist.history['loss'][0]
                    print('Iteration %d : loss = %.4f' % (epoch, loss))

                    if min_loss == None or loss < min_loss:
                        print('Min Loss: ', min_loss)
                        min_loss = loss
                        best_iter = epoch
                        (_, _, top_K_items_scores, rank) = evaluate_instance(model, user_id, validation, test, configs.topK)

            cf_users.append(user_id)
            top_K_items_scores_s.append(top_K_items_scores)
            ranks.append(rank)

            try:
                shift_rank = ranks.index(top1)
            except:
                shift_rank = 20 + 5

            if shift_rank != 0:
                true_cf_count += 1

            shift_rank_s.append(shift_rank)


    if configs.out > 0:
        # Generate report
        col_name = ['user_id_hasCF', 'true_shift_rank', 'top20_index', 'top20_with_score']
        topK_rank_data = {col_name[0]: cf_users, col_name[1]: shift_rank_s ,col_name[2]: ranks, col_name[3]: top_K_items_scores_s}
        df_retrain_rank = pd.DataFrame(topK_rank_data)

        df_retrain_rank.to_csv('Analysis/predict_rank_best_retrain(200-300).csv', index=False, mode='w')
        print('predict_cf_count: ', predict_cf_count)
        print('true_cf_count:', true_cf_count)

