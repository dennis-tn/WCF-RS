from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from collections import namedtuple

from GMF import GMF
from MLP import MLP
from NeuCF import NeuMF

import configs
from DataSet import Dataset
from evaluate import evaluate_model
from utils import get_train_instances, load_pretrain_model

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SparseFeat = namedtuple('SparseFeat', ['name', 'nums', 'mf_dim'])


if __name__ == '__main__':

    weights_out_file = 'Save/%s_NeuMF_%d_%s_ckpt' % (configs.dataset, configs.mf_dim, configs.MLP_layers_units)

    # --------------Loading data--------------
    t1 = time()
    dataset = Dataset('Data/%s' % configs.dataset)

    # Do it with pandas, group, then take the difference set, and then sample 0
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
    # model.build(input_shape=[num_users, num_items]) #[1, 1]

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

    # -----------Load pretrain model-----------
    # Load the previously trained parameters
    if configs.mf_pretrain != '' and configs.mlp_pretrain != '':
        gmf_model = GMF(num_users, num_items, configs.mf_dim)
        gmf_model.build(input_shape=([num_users, num_items]))
        gmf_model.load_weights(configs.mf_pretrain)
        mlp_model = MLP(num_users, num_items, configs.MLP_layers_units, configs.reg_layers)
        mlp_model.build(input_shape=([num_users, num_items]))
        mlp_model.load_weights(configs.mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model, len(configs.MLP_layers_units))

    # ---------------Init performance----------------
    (hits, ndcgs, top_K_items_scores_s, ranks) = evaluate_model(model, validation, test, configs.topK)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f s]' % (hr, ndcg, time() - t1))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if configs.out > 0:
        model.save_weights(weights_out_file, overwrite=True)

    # -----------------Training model-------------
    for epoch in range(configs.epochs):
        t1 = time()
        # Generate training instances
        # pos:neg 1:num_negatives
        user_input, item_input, labels = get_train_instances(df_group_train, num_items, test, configs.num_negatives)
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
        t2 = time()

        # ----------------Evaluation--------------
        if epoch % configs.verbose == 0:
            (hits, ndcgs, top_K_items_scores_s, ranks) = evaluate_model(model, validation, test, configs.topK)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            print('Top %d recommeded items and scores for user_0: %s' % (configs.topK, top_K_items_scores_s[0]))

            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if configs.out > 0:
                    # Save model and weights
                    model.save_weights(weights_out_file, overwrite=True)
                    # model.save(model_out_file, overwrite=True)

                    # Generate report
                    users = list(range(len(top_K_items_scores_s)))
                    col_name = ['user_id', 'top_%s_with_score' % (str(configs.topK),), 'top_%s' % (str(configs.topK),), 'hit_ratio', 'ndcg']
                    topK_rank_data = {col_name[0]: users, col_name[1]: top_K_items_scores_s, col_name[2]: ranks,
                                      col_name[3]: hits, col_name[4]: ndcgs}
                    df_rank = pd.DataFrame(topK_rank_data)

                    df_rank.to_csv('Analysis/predict_rank_best.csv', index=False, mode='w')

    print('End. Best Iteration %d: HR = %.4f, NDCG = %.4f ' % (best_iter, best_hr, best_ndcg))
    if configs.out > 0:
        print('The best NeuMF model is saved to %s' % weights_out_file)


