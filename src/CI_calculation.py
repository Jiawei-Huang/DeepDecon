import collections
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from train_model import *


def get_index(y, diff_mins, diff_maxs, cat, alpha=80):
    upper = y + np.percentile(diff_maxs[cat], alpha)
    lower = y + np.percentile(diff_mins[cat], 100 - alpha)

    lower, upper = max(0, int(10*lower)), min(10, int(10*upper))
    # if upper <= lower:
    #     upper = lower + 1
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
    return min_limit, max_limit

def one_iteration(x, models, idfs, individual, diff_mins, diff_maxs, cat):
    tmp_x, _ = preprocess(x.reshape(1, -1), idfs[cat][individual])
    y_hat = models[cat][individual].predict(tmp_x)
    lower, upper, new_cat = get_index(y_hat[0, 0], diff_mins, diff_maxs, cat)
    return lower, upper, new_cat, y_hat

def get_single_prediction(x, models, idfs, individual, diff_mins, diff_maxs):
    pre_lower, pre_upper, pre_cat = 0, 10, '0_100'
    lower, upper, cat, y_hat = one_iteration(x, models, idfs, individual, diff_mins, diff_maxs, pre_cat)
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
        
        # interval doesn't change or oscillates left and right , stop
        if  lower>=upper or min(dirs)>1 or (lower == pre_lower and upper == pre_upper) or (lower<=pre_lower and upper >= pre_upper):
            flag=0

        if flag:
            pre_lower, pre_upper, pre_cat, pre_y = lower, upper, cat, y_hat
            lower, upper, cat, y_hat = one_iteration(x, models, idfs, individual, diff_mins, diff_maxs, pre_cat) 
            # print(lower, upper, cat, y_hat)
    return y_hat

def get_prediction(X, models, idfs, individual, diff_mins, diff_maxs):
    preds = []
    # data shape ( should be 2 dimention like (140, 7000)) instead of (140, ) )
    for x in X:
        pred = get_single_prediction(x, models, idfs, individual, diff_mins, diff_maxs)
        preds.append(pred)
        # print('Finish one')
    preds = np.concatenate(preds)
    return preds

def main():
    test_genes = pd.read_csv('./aml_subject_data/common_gene.txt', index_col=0)
    keep_gene = ['malignant', 'normal']+list(test_genes['gene'].values)

    subjects = ['AML328-D29', 'AML1012-D0', 'AML556-D0', 'AML328-D171', 
                'AML210A-D0', 'AML419A-D0', 'AML328-D0', 'AML707B-D0',
                'AML916-D0', 'AML328-D113', 'AML329-D0', 'AML420B-D0',
                'AML329-D20', 'AML921A-D0', 'AML475-D0'
            ]

    models = collections.defaultdict(list)
    idfs = collections.defaultdict(list)
    diff_mins = collections.defaultdict(list)
    diff_maxs = collections.defaultdict(list)

    for i in range(15):
        sub = subjects[i]
        m = load_model('./scaden/models/'+sub+'_norm/deepdecon_tf_idf.h5', custom_objects={'rmse':rmse})
        idf = sp.load_npz('./scaden/models/'+sub+'_norm/idf.npz')
        models['0_100'].append(m)
        idfs['0_100'].append(idf)
        path_label = './predictions/prediction_' + sub + '_label1.txt'
        path_pred = './scaden/'+sub+'_norm/'+'prediction_' + sub + '_deepdecon.txt'
        lower, upper = get_difference(path_pred, path_label)
        diff_mins['0_100'].append(lower)
        diff_maxs['0_100'].append(upper)

        for j in range(0, 101, 10):
            k_min = j + 10
            k_max = 100 if j == 0 else 101
            for k in range(k_min, k_max, 10):
                path = './aml_simulated_bulk_data/range_'+ str(j) + '_' + str(k) + '/'
                m = load_model(path+'models/'+sub+'_deepdecon_tf_idf_normalized_m256.h5', custom_objects={'rmse':rmse})
                idf = sp.load_npz(path+'idfs/'+sub+'_normalized_m256.npz')
                models[str(j) + '_' + str(k)].append(m)
                idfs[str(j) + '_' + str(k)].append(idf)
                path_pred = path + 'prediction/' + sub + '_deepdecon_tf_idf_m256_predictions.txt'
                path_data = path + sub + '_bu1lk_nor_500_200.txt'
                lower, upper = get_difference(path_pred, path_data=path_data)
                diff_mins[str(j) + '_' + str(k)].append(lower)
                diff_maxs[str(j) + '_' + str(k)].append(upper)


    ## test on fixed fraction data
    cell = 500
    path_fixed = './aml_bulk_simulation_binomial/'
    for i in range(15):
        sub = subjects[i]
        for ratio in range(0, 100, 10):
            val = pd.read_csv(path_fixed+sub+'_norm/fixed_'+str(ratio)+'_'+str(100-ratio)+'_'+str(cell)+'_1000_binomial.txt', index_col = 0, nrows = 200)
            X_val, y_val = splitData(val[keep_gene])

            pred = get_prediction(X_val.values, models, idfs, i, diff_mins, diff_maxs)
            pred_path = path_fixed+sub+'_norm/predictions/'
            if not os.path.exists(pred_path):
                os.makedirs(pred_path) 

            pd.DataFrame(pred, columns=['malignant', 'normal']).to_csv(pred_path+'fixed_'+str(ratio)+'_'+str(100-ratio)+'_'+str(cell)+'_binomial_prediction.txt')


    ## test and plot figures
    path_fixed = './aml_bulk_simulation_binomial/'
    preds = collections.defaultdict(list)
    labels = collections.defaultdict(list)
    datas = collections.defaultdict(list)
    cell = 500

    for i in range(15):
        sub = subjects[i]
        for ratio in range(0, 100, 10):
            val = pd.read_csv(path_fixed+sub+'_norm/fixed_'+str(ratio)+'_'+str(100-ratio)+'_'+str(cell)+'_1000_binomial.txt', index_col = 0, nrows = 200)
            X_val, y_val = splitData(val[keep_gene])
            
            pred = pd.read_csv(path_fixed+sub+'_norm/predictions/fixed_'+str(ratio)+'_'+str(100-ratio)+'_'+str(cell)+'_binomial_prediction.txt', index_col = 0).values
            preds[str(ratio)].append(pred[:, 0])
            labels[str(ratio)].append(y_val.values[:, 0])
            datas[str(ratio)].append(val[keep_gene])

            metric = np.sqrt(np.mean((y_val.values[:,0]-pred[:, 0])**2))
            print(sub, 'fraction %d%%'%ratio, 'RMSE: %.4f'%metric)
            # plt.figure(figsize=(4,4))
            # plt.plot(y_val.values[:,0], pred[:, 0], '.')
            # plt.plot([0, 1], [0, 1])
            # plt.xlabel('truth normal')
            # plt.ylabel('prediction normal')
            # plt.show()
    subjects = ['AML328-D29', 'AML1012-D0', 'AML556-D0', 'AML328-D171', 
                'AML210A-D0', 'AML419A-D0', 'AML328-D0', 'AML707B-D0',
                'AML916-D0', 'AML328-D113', 'AML329-D0', 'AML420B-D0',
                'AML329-D20', 'AML921A-D0', 'AML475-D0'
            ]
    
if __name__ == "__main__":
    main()
