import os
import pandas as pd
import numpy as np
import argparse
import pickle
import collections
import tensorflow as tf
import scipy.sparse as sp
from util import splitData, preprocess
from tensorflow.keras.models import load_model

def get_index(y, diff_mins, diff_maxs, cat, alpha=10, diff=None):
    if diff is None:
        upper = y + np.percentile(diff_maxs[cat], 100-alpha)
        lower = y + np.percentile(diff_mins[cat], alpha)
    else:
        upper = y + np.percentile(diff[cat], 100 - alpha/2)
        lower = y + np.percentile(diff[cat], alpha/2)
    lower, upper = max(0, int(10*lower)), min(10, int(10*upper))
        
    if upper <= lower:
        upper = lower + 1
    cat = str(lower*10) + '_' + str(upper*10)
    return lower, upper, cat

def get_difference(path_pred, path_label=None, path_data=None):
    if path_label is not None:
        y = np.loadtxt(path_label, skiprows=1, delimiter=',', usecols=(1,2))
    else:
        y = np.loadtxt(path_data, skiprows=1, delimiter=',', usecols=(1,2))
        y = y/np.sum(y, axis=1).reshape(-1, 1)
    y_hat = np.loadtxt(path_pred, skiprows=1, delimiter=',', usecols=(1,2))
    min_limit, max_limit = min(y[:, 0]-y_hat[:, 0]), max(y[:, 0]-y_hat[:, 0])
    return min_limit, max_limit, y[:, 0]-y_hat[:, 0]

def one_iteration(x, models, idfs, individual, diff_mins, diff_maxs, cat, alpha, diff):
    tmp_x, _ = preprocess(x.reshape(1, -1), idfs[cat][individual])
    y_hat = models[cat].predict(tmp_x)
    lower, upper, new_cat = get_index(y_hat[0, 0], diff_mins, diff_maxs, cat, alpha=alpha, diff=diff)
    return lower, upper, new_cat, y_hat

def get_single_prediction(x, models, idfs, individual, diff_mins, diff_maxs, alpha, diff):
    pre_lower, pre_upper, pre_cat = 0, 10, '0_100'
    lower, upper, cat, y_hat = one_iteration(x, models, idfs, individual, diff_mins, diff_maxs, pre_cat, alpha = alpha, diff= diff)
    dirs = [0, 0]
    pre_y = y_hat
    flag = 1
    # print(lower, upper, cat, y_hat)
    while flag:
        # 1. new interval covered by old interval
        if lower >= pre_lower and upper <= pre_upper:
            flag = 1
        elif upper <= pre_upper and lower <= pre_lower:
            flag = 1
            dirs[0] += 1 # go left
        elif lower >= pre_lower and upper >= pre_upper:
            flag = 1
            dirs[1] += 1 # go right
        
        # interval doesn't change or oscillates between left and right , stop
        if  lower>=upper or min(dirs)>1 or (lower == pre_lower and upper == pre_upper) or (lower<=pre_lower and upper >= pre_upper):
            flag=0

        if flag:
            pre_lower, pre_upper, pre_cat, pre_y = lower, upper, cat, y_hat
            lower, upper, cat, y_hat = one_iteration(x, models, idfs, individual, diff_mins, diff_maxs, pre_cat, alpha=alpha, diff = diff) 
            # print(lower, upper, cat, y_hat)
    return y_hat

def get_prediction(X, models, idfs, individual, diff_mins, diff_maxs, alpha=10, diff=None):
    preds = []
    # data shape ( should be 2 dimention like (140, 7000)) instead of (140, ) )
    for x in X:
        pred = get_single_prediction(x, models, idfs, individual, diff_mins, diff_maxs, alpha=alpha, diff=diff)
        preds.append(pred)
        # print('Finish one')
    preds = np.concatenate(preds)
    return preds

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(y_true, y_pred)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, help="Number of cells to use for each bulk sample.", default=500)
    parser.add_argument("--dir", type=str, help="Training data directory", default='./aml_simulated_bulk_data/')
    parser.add_argument("--filepath", type=str, help="Testing file path", default='./aml_simulated_bulk_data/sample_500/range_0_100/AML328-D29_bulk_nor_500_200.txt')
    parser.add_argument("--sub_idx", type=int, help="Testing subject index, 0-14 refers to subjects in the training datasets, 15 means new dataset.", default=0)
    args = parser.parse_args()

    cell = args.cells
    file = args.filepath
    dir = args.dir + 'sample_' + str(cell) + '/'
    sub_idx = args.sub_idx
    test_genes = pd.read_csv('./aml_subject_data/common_gene.txt', index_col=0)
    keep_gene = ['malignant', 'normal']+list(test_genes['gene'].values)


    # loading saved configurations and models
    with open(dir + 'range_results/diff_maxs.pkl', 'rb') as f:
        diff_maxs = pickle.load(f)
    with open(dir + 'range_results/diff_mins.pkl', 'rb') as f:
        diff_mins = pickle.load(f)
    with open(dir + 'range_results/idfs.pkl', 'rb') as f:
        idfs = pickle.load(f)
    with open(dir + 'range_results/merge_diff.pkl', 'rb') as f:
        diff = pickle.load(f)
    models = {}
    # subjects order corresponds to the order in the training process
    subjects = ['AML328-D29', 'AML1012-D0', 'AML556-D0', 'AML328-D171', 
                'AML210A-D0', 'AML419A-D0', 'AML328-D0', 'AML707B-D0',
                'AML916-D0', 'AML328-D113', 'AML329-D0', 'AML420B-D0',
                'AML329-D20', 'AML921A-D0', 'AML475-D0'
            ]

    sub = subjects[sub_idx] if sub_idx < 15 else 'whole'
    for j in range(0, 100, 10):
        for k in range(j+10, 101, 10):
            model_path = dir + 'range_'+ str(j) + '_' + str(k) + '/'
            m = load_model(model_path+'models/'+sub+'_deepdecon_tf_idf_normalized_m256.h5', custom_objects={'rmse':rmse})
            models[str(j) + '_' + str(k)] = m
    print(models.keys())

    
    # load testing data
    val = pd.read_csv(file, index_col = 0, nrows = 200)
    X_val, y_val = splitData(val[keep_gene])

    pred = get_prediction(X_val.values, models, idfs, sub_idx, diff_mins, diff_maxs)
    pred_path = dir+'prediction_'+ str(cell) +'model/'
    if not os.path.exists(pred_path):
        os.makedirs(pred_path) 

    pd.DataFrame(pred, columns=['malignant', 'normal']).to_csv(pred_path+sub+'_deepdecon_tf_idf_m256_predictions.txt')