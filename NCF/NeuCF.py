# -*- coding: utf-8 -*-
# @Data: 2021/10/19
# @Author: Ning TANG

from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers

# from GMF import GMF
# from MLP import MLP
import configs
from DataSet import Dataset
from evaluate import evaluate_model
from utils import get_train_instances, load_pretrain_model

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# cpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


class NeuMF(keras.Model):
    def __init__(self, sparse_feature_dict, MLP_layers_units, reg_layers, reg_mf):
        """
        :param sparse_feature_dict: A list. user feature columns + item feature columns
        :param hidden_layers: A list.
        :param reg_layers:
        :param reg_mf:
        """
        super(NeuMF, self).__init__()
        self.sparse_feature_dict = sparse_feature_dict
        self.MLP_layers_units = MLP_layers_units

        self.MF_Embedding_User = keras.layers.Embedding(
            input_dim=self.sparse_feature_dict['user'].nums,
            output_dim=self.sparse_feature_dict['user'].mf_dim,
            name='mf_embedding_user',
            embeddings_initializer='random_uniform',
            embeddings_regularizer=regularizers.l2(reg_mf[0]),
        )
        self.MF_Embedding_Item = keras.layers.Embedding(
            input_dim=self.sparse_feature_dict['item'].nums,
            output_dim=self.sparse_feature_dict['item'].mf_dim,
            name='mf_embedding_item',
            embeddings_initializer='random_uniform',
            embeddings_regularizer=regularizers.l2(reg_mf[1]),
        )
        self.MLP_Embedding_User = keras.layers.Embedding(
            input_dim=self.sparse_feature_dict['user'].nums,
            output_dim=int(self.MLP_layers_units[0] / 2),
            name='mlp_embedding_user',
            embeddings_initializer='random_uniform',
            embeddings_regularizer=regularizers.l2(reg_layers[0]),
        )
        self.MLP_Embedding_Item = keras.layers.Embedding(
            input_dim=self.sparse_feature_dict['item'].nums,
            output_dim=int(self.MLP_layers_units[0] / 2),
            name='mlp_embedding_item',
            embeddings_initializer='random_uniform',
            embeddings_regularizer=regularizers.l2(reg_layers[0]),
        )


        self.MLP_layers = []
        for i, units in enumerate(MLP_layers_units, start=1):
            self.MLP_layers.append(keras.layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(reg_layers[i]))
            )

        self.NeuMF_layer = keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer='lecun_uniform',
            name='prediction'
        )

    @tf.function
    def call(self, inputs):
        # Embedding
        # Input: n rows and two columns, the first column is user, and the second column is item
        MF_Embedding_User = self.MF_Embedding_User(inputs[:, 0])
        MF_Embedding_Item = self.MF_Embedding_Item(inputs[:, 1])
        MLP_Embedding_User = self.MLP_Embedding_User(inputs[:, 0])
        MLP_Embedding_Item = self.MLP_Embedding_Item(inputs[:, 1])

        # MF
        mf_user_latent = keras.layers.Flatten()(MF_Embedding_User)
        mf_item_latent = keras.layers.Flatten()(MF_Embedding_Item)
        mf_vector = tf.multiply(mf_user_latent, mf_item_latent)

        # MLP
        mlp_user_latent = keras.layers.Flatten()(MLP_Embedding_User)
        mlp_item_latent = keras.layers.Flatten()(MLP_Embedding_Item)

        mlp_vector = tf.concat([mlp_user_latent, mlp_item_latent], axis=1)

        for layer in self.MLP_layers:
            mlp_vector = layer(mlp_vector)

        # The type of emb is int64, and the type after dnn is float32, otherwise an error will be reported
        mf_vector = tf.cast(mf_vector, tf.float32)
        mlp_vector = tf.cast(mlp_vector, tf.float32)

        # NeuMF
        ncf_vector = tf.concat([mf_vector, mlp_vector], axis=1)
        output = self.NeuMF_layer(ncf_vector)

        return output