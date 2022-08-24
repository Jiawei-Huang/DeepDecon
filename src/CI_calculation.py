import collections
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from train_model import *
import random
from rnasieve.preprocessing import model_from_raw_counts


def get_index(y, diff_mins, diff_maxs, cat, alpha=10, diff=None):
    if diff is None:
        upper = y + np.percentile(diff_maxs[cat], 100-alpha)
        lower = y + np.percentile(diff_mins[cat], alpha)
    else:
        upper = y + np.percentile(diff[cat], 100 - alpha/2)
        lower = y + np.percentile(diff[cat], alpha/2)
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
    return min_limit, max_limit, y[:, 0]-y_hat[:, 0]

def one_iteration(x, models, idfs, individual, diff_mins, diff_maxs, cat, alpha, diff):
    tmp_x, _ = preprocess(x.reshape(1, -1), idfs[cat][individual])
    y_hat = models[cat][individual].predict(tmp_x)
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
        
        # interval doesn't change or oscillates left and right , stop
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

def calculate_coverage_rna_sieve(data, cell_onto_class, frac, sig=.05):
    total = 1000
    cnt = 0
    i = 0
    while i <= total:
        ind = random.randint(0, 14)
        if ind == 1 or ind == 8:
            cnt += 2
            i += 2
            continue
        X_val = data[ind]
        sample = X_val.sample(2).T.values
        model, cleaned_psis = model_from_raw_counts(cell_onto_class, sample)
        pred = model.predict(cleaned_psis)
        cis = model.compute_marginal_confidence_intervals(sig=sig)
        ci = [s[0] for s in cis]
   
        for lower, upper in ci:
            if np.isnan(lower) or np.isnan(upper) or (lower <= frac/100 and upper >= frac/100):
                cnt += 1
            i += 1
        if i%100==0:
            print('%d th simulation, Count %d, Coverage %.4f'%(i, cnt, cnt/i))
    print('Coverage: %.4f'%(cnt/(total)))
    return cnt/(total)

def get_coverage(X, model_type, models=None, idfs=None, individual=None, diff_mins=None, diff_maxs=None, counts_onto_class=None):
    # used to test coverage of the model under different fixed fractions
    pass

def main():
    test_genes = pd.read_csv('./aml_subject_data/common_gene.txt', index_col=0)
    keep_gene = ['malignant', 'normal']+list(test_genes['gene'].values)

    subjects = ['AML328-D29', 'AML1012-D0', 'AML556-D0', 'AML328-D171', 
                'AML210A-D0', 'AML419A-D0', 'AML328-D0', 'AML707B-D0',
                'AML916-D0', 'AML328-D113', 'AML329-D0', 'AML420B-D0',
                'AML329-D20', 'AML921A-D0', 'AML475-D0'
            ]

    malig_raw = np.loadtxt('./RNA-sieve/raw_counts_malignant.txt', delimiter=',')
    nor_raw = np.loadtxt('./RNA-sieve/raw_counts_normal.txt', delimiter=',')
    cell_onto_class = {
        'malignant':malig_raw[:, np.random.choice(malig_raw.shape[1], 200)],
        'normal':nor_raw[:, np.random.choice(nor_raw.shape[1], 200)]
    }

    
    cell = 500
    path_fixed = './aml_bulk_simulation_binomial/'
    alpha=0.20
    ratio = 10
    data = []
    for i in range(15):
        sub = subjects[i]
        val = pd.read_csv(path_fixed+sub+'_norm/fixed_'+str(ratio)+'_'+str(100-ratio)+'_'+str(cell)+'_1000_binomial.txt', index_col = 0, nrows = 200)
        X_val, y_val = splitData(val[keep_gene])
        data.append(X_val)
    
    cov = calculate_coverage_rna_sieve(data, cell_onto_class, ratio, alpha)
    print('ratio: %d, alpha: %.4f, coverage: %.4f'%(ratio, alpha, cov))
    cov_path = './RNA-sieve/coverages/'+str(100*(1-alpha))+'/'
    if not os.path.exists(cov_path):
        os.makedirs(cov_path)
    np.savetxt(cov_path + 'coverage_'+str(ratio)+'.txt', np.array([cov]), fmt='%.4f')

        # coverage.append(cov)
        # print('Coverage for %d: %.4f'%(ratio, cov))
    
    # pd.DataFrame(coverage, columns=['coverage'], index=range(10, 100, 10)).to_csv('./RNA-sieve/fixed_prediction/coverage_'+str(1-100*alpha)+'_skip.csv')



    # models = collections.defaultdict(list)
    # idfs = collections.defaultdict(list)
    # diff_mins = collections.defaultdict(list)
    # diff_maxs = collections.defaultdict(list)

    # for i in range(15):
    #     sub = subjects[i]
    #     m = load_model('./scaden/models/'+sub+'_norm/deepdecon_tf_idf.h5', custom_objects={'rmse':rmse})
    #     idf = sp.load_npz('./scaden/models/'+sub+'_norm/idf.npz')
    #     models['0_100'].append(m)
    #     idfs['0_100'].append(idf)
    #     path_label = './predictions/prediction_' + sub + '_label1.txt'
    #     path_pred = './scaden/'+sub+'_norm/'+'prediction_' + sub + '_deepdecon.txt'
    #     lower, upper = get_difference(path_pred, path_label)
    #     diff_mins['0_100'].append(lower)
    #     diff_maxs['0_100'].append(upper)

    #     for j in range(0, 101, 10):
    #         k_min = j + 10
    #         k_max = 100 if j == 0 else 101
    #         for k in range(k_min, k_max, 10):
    #             path = './aml_simulated_bulk_data/range_'+ str(j) + '_' + str(k) + '/'
    #             m = load_model(path+'models/'+sub+'_deepdecon_tf_idf_normalized_m256.h5', custom_objects={'rmse':rmse})
    #             idf = sp.load_npz(path+'idfs/'+sub+'_normalized_m256.npz')
    #             models[str(j) + '_' + str(k)].append(m)
    #             idfs[str(j) + '_' + str(k)].append(idf)
    #             path_pred = path + 'prediction/' + sub + '_deepdecon_tf_idf_m256_predictions.txt'
    #             path_data = path + sub + '_bu1lk_nor_500_200.txt'
    #             lower, upper = get_difference(path_pred, path_data=path_data)
    #             diff_mins[str(j) + '_' + str(k)].append(lower)
    #             diff_maxs[str(j) + '_' + str(k)].append(upper)
                

    # ## test on fixed fraction data
    # cell = 500
    # path_fixed = './aml_bulk_simulation_binomial/'
    # for i in range(15):
    #     sub = subjects[i]
    #     for ratio in range(0, 100, 10):
    #         val = pd.read_csv(path_fixed+sub+'_norm/fixed_'+str(ratio)+'_'+str(100-ratio)+'_'+str(cell)+'_1000_binomial.txt', index_col = 0, nrows = 200)
    #         X_val, y_val = splitData(val[keep_gene])

    #         pred = get_prediction(X_val.values, models, idfs, i, diff_mins, diff_maxs)
    #         pred_path = path_fixed+sub+'_norm/predictions/'
    #         if not os.path.exists(pred_path):
    #             os.makedirs(pred_path) 

    #         pd.DataFrame(pred, columns=['malignant', 'normal']).to_csv(pred_path+'fixed_'+str(ratio)+'_'+str(100-ratio)+'_'+str(cell)+'_binomial_prediction.txt')


    # ## test and plot figures
    # path_fixed = './aml_bulk_simulation_binomial/'
    # preds = collections.defaultdict(list)
    # labels = collections.defaultdict(list)
    # datas = collections.defaultdict(list)
    # cell = 500

    # for i in range(15):
    #     sub = subjects[i]
    #     for ratio in range(0, 100, 10):
    #         val = pd.read_csv(path_fixed+sub+'_norm/fixed_'+str(ratio)+'_'+str(100-ratio)+'_'+str(cell)+'_1000_binomial.txt', index_col = 0, nrows = 200)
    #         X_val, y_val = splitData(val[keep_gene])
            
    #         pred = pd.read_csv(path_fixed+sub+'_norm/predictions/fixed_'+str(ratio)+'_'+str(100-ratio)+'_'+str(cell)+'_binomial_prediction.txt', index_col = 0).values
    #         preds[str(ratio)].append(pred[:, 0])
    #         labels[str(ratio)].append(y_val.values[:, 0])
    #         datas[str(ratio)].append(val[keep_gene])

    #         metric = np.sqrt(np.mean((y_val.values[:,0]-pred[:, 0])**2))
    #         print(sub, 'fraction %d%%'%ratio, 'RMSE: %.4f'%metric)
            # plt.figure(figsize=(4,4))
            # plt.plot(y_val.values[:,0], pred[:, 0], '.')
            # plt.plot([0, 1], [0, 1])
            # plt.xlabel('truth normal')
            # plt.ylabel('prediction normal')
            # plt.show()
    
if __name__ == "__main__":
    # main()
    os.chdir('/project/fsun_106/jiaweih/AML/10x_dataset/gdc_data/')
    test_genes = pd.read_csv('./aml_subject_data/common_gene.txt', index_col=0)
    keep_gene = ['malignant', 'normal']+list(test_genes['gene'].values)

    subjects = ['AML328-D29', 'AML1012-D0', 'AML556-D0', 'AML328-D171', 
                'AML210A-D0', 'AML419A-D0', 'AML328-D0', 'AML707B-D0',
                'AML916-D0', 'AML328-D113', 'AML329-D0', 'AML420B-D0',
                'AML329-D20', 'AML921A-D0', 'AML475-D0'
            ]
    cells = 500
    path = './aml_simulated_bulk_data/sample_' + str(cells) + '/'
    # models = collections.defaultdict(list)
    # idfs = collections.defaultdict(list)
    # diff_mins = collections.defaultdict(list)
    # diff_maxs = collections.defaultdict(list)
    # difference = collections.defaultdict(list)

    # for i in range(15):
    #     sub = subjects[i]
    #     for j in range(0, 100, 10):
    #         for k in range(j+10, 101, 10):
    #             # dir = path+'range_'+str(i) + '_' + str(j) + '/prediction/' + sub + '_deepdecon_tf_idf_m256_predictions.txt'
    #             dir = path + 'range_'+ str(j) + '_' + str(k) + '/'
    #             # m = load_model(path+'models/'+sub+'_deepdecon_tf_idf_normalized_m256.h5', custom_objects={'rmse':rmse})
    #             # idf = sp.load_npz(dir+'idfs/'+sub+'_normalized_m256_test.npz')
    #             # models[str(j) + '_' + str(k)].append(m)
    #             # idfs[str(j) + '_' + str(k)].append(idf)
    #             path_pred = dir + 'prediction/' + sub + '_deepdecon_tf_idf_m256_predictions.txt'
    #             path_data = dir + sub + '_bulk_nor_'+ str(cells) +'_200.txt'
    #             lower, upper, diffs = get_difference(path_pred, path_data=path_data)
    #             # diff_mins[str(j) + '_' + str(k)].append(lower)
    #             # diff_maxs[str(j) + '_' + str(k)].append(upper)
    #             difference[str(j) + '_' + str(k)].append(diffs)

    import pickle
    # res_path = path + 'range_results/'
    # if not os.path.exists(res_path):
    #     os.makedirs(res_path)

    # with open(res_path + 'diff_maxs.pkl', 'wb') as f:
    #     pickle.dump(diff_maxs, f)
    # with open(res_path + 'diff_mins.pkl', 'wb') as f:
    #     pickle.dump(diff_mins, f)
    # with open(res_path + 'idfs.pkl', 'wb') as f:
    #     pickle.dump(idfs, f)
    # with open(res_path + 'difference.pkl', 'wb') as f:
    #     pickle.dump(difference, f)
    

    with open(path + 'range_results/diff_maxs.pkl', 'rb') as f:
        diff_maxs = pickle.load(f)
    with open(path + 'range_results/diff_mins.pkl', 'rb') as f:
        diff_mins = pickle.load(f)
    with open(path + 'range_results/idfs.pkl', 'rb') as f:
        idfs = pickle.load(f)
    with open(path + 'range_results/merge_diff.pkl', 'rb') as f:
        diff = pickle.load(f)
    models = collections.defaultdict(list)

    for i in range(15):
        sub = subjects[i]
        for j in range(0, 100, 10):
            for k in range(j+10, 101, 10):
                model_path = path + 'range_'+ str(j) + '_' + str(k) + '/'
                m = load_model(model_path+'models/'+sub+'_deepdecon_tf_idf_normalized_m256.h5', custom_objects={'rmse':rmse})
                models[str(j) + '_' + str(k)].append(m)
    print(models.keys())

    cells = 3000
    path = './aml_simulated_bulk_data/sample_' + str(cells) + '/'
    i = 8
    sub = subjects[i]
    for j in range(0, 100, 10):
        for k in range(j+10, 101, 10):
            dir = path + 'range_'+ str(j) + '_' + str(k) + '/'
            val = pd.read_csv(dir+sub+'_bulk_nor_'+ str(cells) + '_200.txt', index_col = 0, nrows = 200)
            X_val, y_val = splitData(val[keep_gene])

            pred = get_prediction(X_val.values, models, idfs, i, diff_mins, diff_maxs)
            pred_path = dir+'prediction_500model/'
            if not os.path.exists(pred_path):
                os.makedirs(pred_path) 

            pd.DataFrame(pred, columns=['malignant', 'normal']).to_csv(pred_path+sub+'_deepdecon_tf_idf_m256_predictions.txt')





    # preds = collections.defaultdict(list)
    # labels = collections.defaultdict(list)
    # cells = 3000
    # path_fixed = './aml_bulk_simulation_binomial/sample_' + str(cells) + '/'
    # alpha = 500
    # num = 1000 if cells == 500 else 200
    # for i in range(15):
    #     sub = subjects[i]
    #     for ratio in range(0, 100, 10):
    #         val = pd.read_csv(path_fixed+sub+'_norm/fixed_'+str(ratio)+'_'+str(100-ratio)+'_'+str(cells)+'_'+ str(num)+'_binomial.txt', index_col = 0, nrows = 200)
    #         X_val, y_val = splitData(val[keep_gene])

    #         pred = get_prediction(X_val.values, models, idfs, i, diff_mins, diff_maxs, alpha=10, diff=diff)
    #         pred_path = path_fixed+sub+'_norm/predictions_'+str(alpha) + 'model/'
    #         if not os.path.exists(pred_path):
    #             os.makedirs(pred_path) 

    #         pd.DataFrame(pred, columns=['malignant', 'normal']).to_csv(pred_path+'fixed_'+str(ratio)+'_'+str(100-ratio)+'_'+str(cells)+'_binomial_prediction.txt')
            
            # preds[str(ratio)].append(pred[:, 0])
            # labels[str(ratio)].append(y_val.values[:, 0])


##############################################################################################################################
#  write models
##############################################################################################################################

# def get_difference(path_pred, path_label=None, path_data=None):
#     if path_label is not None:
#         y = np.loadtxt(path_label, skiprows=1, delimiter=',', usecols=(1,2))
#     else:
#         y = np.loadtxt(path_data, skiprows=1, delimiter=',', usecols=(1,2))
#         y = y/np.sum(y, axis=1).reshape(-1, 1)
#     y_hat = np.loadtxt(path_pred, skiprows=1, delimiter=',', usecols=(1,2))
#     min_limit, max_limit = min(y[:, 0]-y_hat[:, 0]), max(y[:, 0]-y_hat[:, 0])
#     return min_limit, max_limit
# models = collections.defaultdict(list)
# idfs = collections.defaultdict(list)
# diff_mins = collections.defaultdict(list)
# diff_maxs = collections.defaultdict(list)

# for i in range(15):
#     sub = subjects[i]
#     for j in range(0, 100, 10):
#         for k in range(j+10, 101, 10):
#             # dir = path+'range_'+str(i) + '_' + str(j) + '/prediction/' + sub + '_deepdecon_tf_idf_m256_predictions.txt'
#             dir = path + 'range_'+ str(j) + '_' + str(k) + '/'
#             # m = load_model(path+'models/'+sub+'_deepdecon_tf_idf_normalized_m256.h5', custom_objects={'rmse':rmse})
#             idf = sp.load_npz(dir+'idfs/'+sub+'_normalized_m256_test.npz')
#             # models[str(j) + '_' + str(k)].append(m)
#             idfs[str(j) + '_' + str(k)].append(idf)
#             path_pred = dir + 'prediction/' + sub + '_deepdecon_tf_idf_m256_predictions.txt'
#             path_data = dir + sub + '_bulk_nor_'+ str(cells) +'_200.txt'
#             lower, upper = get_difference(path_pred, path_data=path_data)
#             diff_mins[str(j) + '_' + str(k)].append(lower)
#             diff_maxs[str(j) + '_' + str(k)].append(upper)
# import pickle
# res_path = path + 'range_results/'
# if not os.path.exists(res_path):
#     os.makedirs(res_path)

# with open(res_path + 'diff_maxs.pkl', 'wb') as f:
#     pickle.dump(diff_maxs, f)
# with open(res_path + 'diff_mins.pkl', 'wb') as f:
#     pickle.dump(diff_mins, f)
# with open(res_path + 'idfs.pkl', 'wb') as f:
#     pickle.load(idfs, f)

# #############################################################################
# read models, fixed data to draw distribution of predictions
#  #############################################################################

# os.chdir('/project/fsun_106/jiaweih/AML/10x_dataset/gdc_data/')
# test_genes = pd.read_csv('./aml_subject_data/common_gene.txt', index_col=0)
# keep_gene = ['malignant', 'normal']+list(test_genes['gene'].values)

# subjects = ['AML328-D29', 'AML1012-D0', 'AML556-D0', 'AML328-D171', 
#             'AML210A-D0', 'AML419A-D0', 'AML328-D0', 'AML707B-D0',
#             'AML916-D0', 'AML328-D113', 'AML329-D0', 'AML420B-D0',
#             'AML329-D20', 'AML921A-D0', 'AML475-D0'
#         ]
# cells = 1000
# path = './aml_simulated_bulk_data/sample_' + str(cells) + '/'
# import pickle 
# with open(path + 'range_results/diff_maxs.pkl', 'rb') as f:
#     diff_maxs = pickle.load(f)
# with open(path + 'range_results/diff_mins.pkl', 'rb') as f:
#     diff_mins = pickle.load(f)
# with open(path + 'range_results/idfs.pkl', 'rb') as f:
#     idfs = pickle.load(f)
# models = collections.defaultdict(list)

# for i in range(15):
#     sub = subjects[i]
#     for j in range(0, 101, 10):
#         for k in range(j+10, 101, 10):
#             model_path = path + 'range_'+ str(j) + '_' + str(k) + '/'
#             m = load_model(model_path+'models/'+sub+'_deepdecon_tf_idf_normalized_m256.h5', custom_objects={'rmse':rmse})
#             models[str(j) + '_' + str(k)].append(m)
# print(models.keys())

# preds = collections.defaultdict(list)
# labels = collections.defaultdict(list)
# cell = 500
# path_fixed = './aml_bulk_simulation_binomial/sample_' + str(cells) + '/'
# for i in range(15):
#     sub = subjects[i]
#     for ratio in range(0, 100, 10):
#         y = np.loadtxt(path_fixed+sub+'_norm/fixed_'+str(ratio)+'_'+str(100-ratio)+'_'+str(cell)+'_1000_binomial.txt', skiprows=1, delimiter=',', usecols=(1,2), max_rows=200)
#         y = y/np.sum(y, axis=1).reshape(-1, 1)

#         y_hat = np.loadtxt(path_fixed+sub+'_norm/predictions/fixed_'+str(ratio)+'_'+str(100-ratio)+'_'+str(cell)+'_binomial_prediction.txt', skiprows=1, delimiter=',', usecols=(1,2))
#         preds[str(ratio)].append(y_hat[:, 0])
#         labels[str(ratio)].append(y[:, 0])

# import seaborn as sns
# fig, axs = plt.subplots(2, 5, figsize=(14,8))
# inds = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
# for i, ax in enumerate(axs.ravel()):
#     pred = np.concatenate([preds[str(i*10)][j] for j in inds])
#     label = np.concatenate([labels[str(i*10)][j] for j in inds])
#     sns.distplot(preds[str(i*10)], ax=ax)
#     ax.get_lines()[0].remove()
#     x = np.linspace(0, 1, 100)
    # phi = f(np.mean(preds[str(i*10)]),-130, 45)
    # ax.plot(x, beta.pdf(x, phi*np.mean(preds[str(i*10)]), phi*(1-np.mean(preds[str(i*10)]))),'y-', lw=2)
#     metric = np.sqrt(np.mean((label-pred)**2))
#     ax.set_title('ratio: %f, RMSE: %.4f'%((i)/10, metric))
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 7)

# from scipy.stats import beta
# import random

# def f(x, a, b):
#     return a*(x-0.5)**2 + b

# total = 1000
# cnt = 0
# alpha = 0.05
# N=500
# coverages = []
# for frac in range(1, 10):
#     print('Fraction', frac/10)
#     cnt = 0
#     for i in range(total):
#         ind = random.randint(0, 14)
#         if ind == 1 or ind == 8:
#             cnt += 1
#             continue
#         m = models[ind]
#         idf = idfs[ind]
#         ind_rand = np.random.randint(len(preds[str(frac*10)][ind]))
#         malig = preds[str(frac*10)][ind][ind_rand]
#         phi = f(malig, -130, 45)
#         shape1, shape2 = phi*malig, phi*(1-malig)
#         lower = beta.ppf(q=alpha/2,a=shape1,b=shape2,loc=0,scale=1)
#         upper = beta.ppf(q=1-alpha/2,a=shape1,b=shape2,loc=0,scale=1)

#         if np.isnan(lower) or np.isnan(upper) or (lower <= frac/10 and upper >= frac/10):
#             cnt += 1
#         if (i+1)%100==0:
#             print('%d th simulation, Count %d, Coverage %.4f'%(i+1, cnt, cnt/(i+1)))

#     print('Coverage: %.4f'%(cnt/total))
#     coverages.append(cnt/total)



# preds_total, labels_total = [], []

# for cells in [500, 1000, 2000, 3000]:
#     path_fixed = './aml_bulk_simulation_binomial/sample_' + str(cells) + '/'
#     preds = collections.defaultdict(list)
#     labels = collections.defaultdict(list)
#     for i in range(15):
#         sub = subjects[i]
#         num = 1000 if cells == 500 else 200
#         for ratio in range(0, 100, 10):
#             y = np.loadtxt(path_fixed+sub+'_norm/fixed_'+str(ratio)+'_'+str(100-ratio)+'_'+str(cells)+'_'+ str(num) +'_binomial.txt', skiprows=1, delimiter=',', usecols=(1,2), max_rows=200)
#             y = y/np.sum(y, axis=1).reshape(-1, 1)

#             y_hat = np.loadtxt(path_fixed+sub+'_norm/predictions/fixed_'+str(ratio)+'_'+str(100-ratio)+'_'+str(cells)+'_binomial_prediction.txt', skiprows=1, delimiter=',', usecols=(1,2))
#             preds[str(ratio)].append(y_hat[:, 0])
#             labels[str(ratio)].append(y[:, 0])
#     print(cells, 'Done!')
#     preds_total.append(preds)
#     labels_total.append(labels)

# vars_all, means_all = [], []
# for preds in preds_total:
#     vars = []
#     means = []
#     for key in preds:
#         tmp = np.concatenate([preds[key][j] for j in inds])
#         var = np.var(tmp)
#         mean = np.mean(tmp)
#         vars.append(var)
#         means.append(mean)
#     vars_all.append(vars)
#     means_all.append(means)

# fig, axs = plt.subplots(2, 1, figsize=(14,8))
# titles = ['Mean', 'Variance']
# datas = [means_all, vars_all]
# labels = ['500', '1000', '2000', '3000']
# x = [i/10 for i in range(10)]
# for i, ax in enumerate(axs.ravel()):
#     for j, val in enumerate(datas[i]):
#         ax.plot(x, val, label = labels[j], alpha=0.6)
#     ax.set_title(titles[i])

#     ax.set_ylabel('value')
#     ax.legend()
# ax.set_xlabel('fraction')
