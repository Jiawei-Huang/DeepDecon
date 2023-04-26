import os
import pandas as pd
import numpy as np
import argparse
import pickle
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
        if lower == 10:
            upper = lower
            lower -= 1
        else:
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
        if  lower>=upper or min(dirs)>=1 or (lower<=pre_lower and upper >= pre_upper):
            flag=0

        if flag:
            pre_lower, pre_upper, pre_cat, pre_y = lower, upper, cat, y_hat
            lower, upper, cat, y_hat = one_iteration(x, models, idfs, individual, diff_mins, diff_maxs, pre_cat, alpha=alpha, diff = diff) 
            # print(lower, upper, cat, y_hat)
    return y_hat

def get_prediction(X, models, idfs, individual, diff_mins=None, diff_maxs=None, alpha=10, diff=None):
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
    parser.add_argument("--outfile", type=str, help="Prediction filename", default='./aml_bulk_simulation_binomial/sample_500/predictions/fixed_10_90_500_binomial_prediction.txt')
    parser.add_argument("--sub_idx", type=int, help="Testing subject index, 0-14 refers to subjects in the training datasets, 15 means new dataset.", default=0)
    # parser.add_argument("--ratio", type=int, help="fixed ratio", default=10)
    args = parser.parse_args()

    cell = args.cells
    file = args.filepath
    outfile = args.outfile
    sub_idx = args.sub_idx

    # load models
    models = {}
    for j in range(0, 100, 10):
        for k in range(j+10, 101, 10):
            model_path = file + 'models/range_'+ str(j) + '_' + str(k) \
                        + '_deepdecon_tf_idf_normalized_m256.h5'
            m = load_model(model_path, custom_objects={'rmse':rmse})
            models[str(j) + '_' + str(k)] = m
    print(models.keys())

    metric_path = file + 'range_results/'
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)
        idfs = {}
        difference = {}

        for j in range(0, 100, 10):
            for k in range(j+10, 101, 10):
                data_path = file + 'data/range_'+ str(j) + '_' + str(k) + '_bulk_nor_500_3000.txt'
                idf_path = file + 'idfs/range_'+ str(j) + '_' + str(k) + '_normalized_m256.npz'
                data = pd.read_csv(data_path, index_col = 0)
                idf = sp.load_npz(idf_path)

                X_val, y_val = splitData(data)
                nor_X_val_scale, _ = preprocess(X_val, idf)
                preds = models[str(j) + '_' + str(k)].predict(nor_X_val_scale)
                diff = y_val.values[:, 0]-preds[:, 0]
                difference[str(j) + '_' + str(k)] = diff
                idfs[str(j) + '_' + str(k)] = [idf]
        
        with open(metric_path + 'idfs.pkl', 'wb') as f:
            pickle.dump(idfs, f)
        with open(metric_path + 'difference.pkl', 'wb') as f:
            pickle.dump(difference, f)
    else:
        # loading saved configurations and models
        with open(metric_path + 'idfs.pkl', 'rb') as f:
            idfs = pickle.load(f)
        with open(metric_path + 'difference.pkl', 'rb') as f:
            difference = pickle.load(f)

    data = pd.read_csv(file + 'data/test_bulk_nor_500_200.txt', index_col = 0)
    X_val, y_val = splitData(data)
    nor_X_val_scale, _ = preprocess(X_val, idfs['0_100'][0])
    pred1 = models['0_100'].predict(nor_X_val_scale)
    pd.DataFrame(pred1, columns=['malignant', 'normal']).to_csv(file+'prediction/deepdecon_prediction.txt')
    pred = get_prediction(X_val.values, models, idfs, 0, diff=difference)

    pred_path = outfile[:outfile.rfind('/')]
    if not os.path.exists(pred_path):
        os.makedirs(pred_path) 

    pd.DataFrame(pred, columns=['malignant', 'normal']).to_csv(outfile)
    print(outfile, 'Finish')