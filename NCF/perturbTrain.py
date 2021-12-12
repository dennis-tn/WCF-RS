from NeuCF import NeuMF

from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers

from GMF import GMF
from MLP import MLP
import configs
from DataSet import Dataset
from evaluate import evaluate_model, evaluate_instance
from utils import sample_per_user

import explainers as ex
from collections import namedtuple
import os

import scipy.sparse as sp

gpus= tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Use a named tuple to define a feature tag: it is composed of a name and a domain,
# which is similar to a dictionary but cannot be changed, which is lightweight and convenient
SparseFeat = namedtuple('SparseFeat', ['name', 'nums', 'mf_dim'])

if __name__ == '__main__':

    weights_out_file = 'Save/%s_NeuMF_%d_%s_ckpt' % (configs.dataset, configs.mf_dim, configs.MLP_layers_units)

    # --------------Loading data--------------
    t1 = time()
    dataset = Dataset('Data/%s' % configs.dataset)
    train, validation, test = dataset.trainMatrix, dataset.valList, dataset.testList
    df_group_train = dataset.df_group_train
    users, num_users = dataset.users, dataset.num_users
    items, num_items = dataset.items, dataset.num_items

    feature_columns_dict = {'user': SparseFeat('user_id', num_users, 8),
                            'item': SparseFeat('item_id', num_items, 8)}

    print('Load data done [%.1f s]. # user=%d, #item=%d, #train=%d, #test=%d'
          % (time()-t1, num_users, num_items, train.nnz, len(validation)))

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

    # -----------Generate perturbed samples-----------
    total_time = []

    for user_id in configs.perturbed_users:
        # Load the parameters after the first training
        model.load_weights(weights_out_file)

        perturbed_samples, sim_user_ids, select_items_num, keep_index_s, masked_index_s, masked_lens = \
            ex.get_perturbed_samples(user_id, configs.num_samples, similarity_size=[3, 5])

        print('finish getting perturbed samples.')
        # -----------Creat CF List-----------
        top_K_items_scores_s, ranks = [],[]

        # -----------------Training model-------------
        total_input = []
        total_train = []

        total_p_start = time()

        for i in range(configs.num_samples):
            model.load_weights(weights_out_file)
            min_loss=None
            print('perturbed_samples ', i)
            # cf_inputs.append(perturbed_samples[i])
            epoch_input = []
            epoch_train = []
            # epoch_eval = []
            for epoch in range(configs.retrain_epochs):
                t1 = time()
                # Generate training instances
                # pos:neg 1:num_negatives
                retrain_user_input, retrain_item_input, retrain_labels = ex.generate_retrain_data(user_id,
                                                                                                  keep_index_s[i],
                                                                                                  masked_index_s[i],
                                                                                                  sim_user_ids,
                                                                                                  df_group_train,
                                                                                                  num_items,
                                                                                                  test)
                # print('retrain_user_input', retrain_user_input[:3])
                # print('retrain_item_input', retrain_item_input[:3])
                print('Len of retrain user input: ', len(retrain_user_input))
                get_input_time = round(time() - t1,2)
                # Get the data time of each epoch
                epoch_input.append(get_input_time)
                print('Get_train_instances [%s]' % (get_input_time,))

                # --------------Fit model----------------
                t1 = time()
                X = np.array([retrain_user_input, retrain_item_input]).T

                hist = model.fit(X,
                                 np.array(retrain_labels),
                                 batch_size=configs.batch_size,
                                 epochs=1,
                                 verbose=configs.verbose,
                                 shuffle=True)
                get_train_time = round(time() - t1, 2)
                # Get the training time of each epoch
                epoch_train.append(get_train_time)
                print('Iteration %d [%s]' % (epoch, get_train_time))

                loss = hist.history['loss'][0]
                if min_loss==None or loss < min_loss:
                    min_loss = loss
                    best_iter = epoch
                    (_, _, top_K_items_scores, ranklist) = evaluate_instance(model, user_id, validation, test,
                                                                                 configs.topK)


            # Average time after one epoch
            avg_get_input = round(np.mean(epoch_input), 2)
            avg_train = round(np.mean(epoch_train), 2)

            total_input.append(avg_get_input)
            total_train.append(avg_train)


            # ----------------Evaluation--------------
            t1 = time()

            print('User %d : best_epoch = %d, loss = %.4f' % (user_id, best_iter, min_loss))
            print('Top %d recommeded items and scores for user_%d: %s' % (configs.topK, user_id, top_K_items_scores))

            top_K_items_scores_s.append(top_K_items_scores)
            ranks.append(ranklist)

        # The total perturbed training time of a sample
        total_p = round(time() - total_p_start, 2)
        # mean time
        avg_p = round(total_p/configs.num_samples, 2)

        avg_total_input = round(np.mean(total_input), 2)
        avg_total_train = round(np.mean(total_train), 2)

        total_time.append(total_p)

        print('User %d | %d perturbed samples | total time : [%s] | mean time : [%s]' % (user_id, configs.num_samples, total_p, avg_p))
        print('For every perturbed sample : mean get input time : [%s] | mean train time : [%s]' % (avg_total_input, avg_total_train))

        # Generate report
        col_name = ['perturbed_top_10', 'perturbed_rank', 'keep_len', 'keep_index',
                    'masked_len','masked_index']
        topK_rank_data = {col_name[0]: top_K_items_scores_s,
                          col_name[1]: ranks,
                          col_name[2]: select_items_num,
                          col_name[3]: keep_index_s,
                          col_name[4]: masked_lens,
                          col_name[5]: masked_index_s
                          }


        df_rank = pd.DataFrame(topK_rank_data)

        df_rank.to_csv('Analysis/Perturbed_Data/perturbed_rank_user_%d_sample_%d.csv'%(user_id, configs.num_samples), index=False, mode='w')

        sp.save_npz('Analysis/Perturbed_Data/perturbed_samples_user_%d_sample_%d.npz'%(user_id,configs.num_samples), perturbed_samples.tocsr())

    np.save('Analysis/total_time_(200-300).npy', np.array(total_time))




