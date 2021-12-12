import numpy as pd
import pandas as pd
import rbo
import configs

def assign_label(row, cur_user_top1):
    row = eval(row)
    ptop1 = row[0]
    if ptop1 == cur_user_top1:
        return 0
    else:
        return 1

def get_shift_rank(p_rank, cur_user_top1, k):
    p_rank = eval(p_rank)[:k]
    try:
        return p_rank.index(cur_user_top1)
    except:
        return k+5

def getJaccard(p_rank, o_rank, k):
    p_rank = eval(p_rank)[:k]
    inter = len(set(o_rank)&set(p_rank))
    uni = len(set(o_rank)|set(p_rank))
    return inter/uni

def getRbo(p_rank, o_rank,  k):
    p_rank = eval(p_rank)[:k]
    return(rbo.RankingSimilarity(o_rank, p_rank).rbo())

df_original = pd.read_csv('Analysis/predict_rank_best.csv')

# top-K rec List
topK = [5, 10, 20]


for user_id in configs.perturbed_users:
    # top20 list
    cur_user_top20 = eval(df_original.at[user_id, 'top_20'])  # [df_original['user_id']==user_id]['top_10'][0]
    cur_user_top1 = int(cur_user_top20[0])

    df_perturb = pd.read_csv('Analysis/Perturbed_Data/perturbed_rank_user_%d_sample_%d.csv' % (user_id, configs.num_samples))

    df_perturb['top1_change'] = df_perturb['perturbed_rank'].apply(assign_label, args=(cur_user_top1,))

    for k in topK:
        topK_ori = cur_user_top20[:k]
        _jaccard = 'Top%d_Jaccard'%(k,)
        _rbo = 'Top%d_Rbo'%(k,)
        _shift_rank = 'Top%d_ShiftRank'%(k,)
        df_perturb[_jaccard] = df_perturb['perturbed_rank'].apply(getJaccard, args=(topK_ori,k))
        df_perturb[_rbo] = df_perturb['perturbed_rank'].apply(getRbo, args=(topK_ori,k))
        df_perturb[_shift_rank] = df_perturb['perturbed_rank'].apply(get_shift_rank, args=(cur_user_top1, k))


    df_perturb.to_csv('Analysis/CF_Data/perturbed_rank_user_%d_sample_%d.csv' % (user_id,configs.num_samples), index=False)
