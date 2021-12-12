# -*- coding: utf-8 -*-
# @Data: 2021/10/19
# @Author: Ning TANG

# General Arguments
dataset = 'ml-100k/ml-100k'
verbose = 1
topK = 20
out = 1


# NeuMF

epochs = 10
batch_size = 256
mf_dim = 8
reg_mf = [1e-3, 1e-3]
MLP_layers_units = [32, 16, 8]
reg_layers = [1e-3, 1e-3, 1e-3, 1e-3]
num_negatives = 4
learning_rate = 0.001
learner = 'adam'
num_factors = 8
mf_pretrain = ''        # 'Pretrain/ml-1m_GMF_8_1575894502.h5'
mlp_pretrain = ''       # 'Pretrain/ml-1m_MLP_[64, 32, 16, 8]_1575898018.h5'


retrain_epochs = 5
num_samples=300

perturbed_users = list(range(200,300))

retrain_init = 'ml-100k_cf_top3/ml-100k_cf_top3'

"""
# GMF
epochs = 2
batch_size = 256
num_negatives = 4
num_factors = 8
learner = 'adam'
learning_rate = 0.001
regs = [0, 0]
"""

"""
# MLP
epochs = 2
batch_size = 256
layers = [64, 32, 16, 8]
reg_layers = [0, 0, 0, 0]
num_negatives = 4
learning_rate = 0.001
learner = 'adam'
"""