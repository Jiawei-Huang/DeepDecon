import collections
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from train_model import *


def get_index(x, diffs):
    index =  int(10*x)
    if x < 0.5:
        index = min(4, int(10*x+8*diffs[index]))
    else:
        index = max(5, int(10*x-8*diffs[index]))
    
    if index<5:
        return index, 'less_'+str((index+1)*10)
    else:
        return index, 'greater_'+str(index*10)

def get_single_prediction(x, models, idfs, individual, diffs):
    pre_cat = 'full'
    tmp_x, _ = preprocess(x.reshape(1, -1), idfs[pre_cat][individual])
    y_hat = models[pre_cat][individual].predict(tmp_x)
    index, cat = get_index(y_hat[0, 0], diffs)
    pre_index = 5 if index<5 else 4
    cnt = [0, 0]

    while (y_hat[0,0]<0.5 and index<pre_index) or (y_hat[0,0]>0.5 and index>pre_index):
        pre_cat = cat
        pre_index = index
        if y_hat[0,0] < 0.5:
            cnt[0] += 1
        else:
            cnt[1] += 1
        if min(cnt)>1:
            break
        tmp_x, _ = preprocess(x.reshape(1, -1), idfs[pre_cat][individual])
        y_hat = models[pre_cat][individual].predict(tmp_x)
        index, cat = get_index(y_hat[0, 0], diffs)
    
    return y_hat

def get_prediction(X, models, idfs, individual, diffs):
    preds = []
    print('data shape ( should be 2 dimention like (140, 7000)) instead of (140, ) ):', X.shape)
    for x in X:
        pred = get_single_prediction(x, models, idfs, individual, diffs)
        preds.append(pred)
    preds = np.concatenate(preds)
    return preds

test_genes = pd.read_csv('./aml_subject_data/common_gene.txt', index_col=0)
keep_gene = ['malignant', 'normal']+list(test_genes['gene'].values)

subjects = ['AML328-D29', 'AML1012-D0', 'AML556-D0', 'AML328-D171', 
            'AML210A-D0', 'AML419A-D0', 'AML328-D0', 'AML707B-D0',
            'AML916-D0', 'AML328-D113', 'AML329-D0', 'AML420B-D0',
            'AML329-D20', 'AML921A-D0', 'AML475-D0'
           ]
diffs = [0.08462659, 0.16956725, 0.1962019, 0.2492649, 0.2951546,
       0.33725176, 0.21845662, 0.18329535, 0.13699717, 0.07919444]

models = collections.defaultdict(list)
idfs = collections.defaultdict(list)

path_fixed = './aml_bulk_simulation_binomial/'
for i in range(15):
    sub = subjects[i]
    m = load_model('./scaden/models/'+sub+'_norm/deepdecon_tf_idf.h5', custom_objects={'rmse':rmse})
    idf = sp.load_npz('./scaden/models/'+sub+'_norm/idf.npz')
    models['full'].append(m)
    idfs['full'].append(idf)
    for j in range(10):
        if j < 5:
            cat = 'less_'+str((j+1)*10)
        else:
            cat = 'greater_'+str(j*10)

        path = './aml_simulated_bulk_data/AML_'+cat+'/'
        m = load_model(path+'models/'+sub+'_deepdecon_tf_idf_normalized_m256.h5', custom_objects={'rmse':rmse})
        idf = sp.load_npz(path+'idfs/'+sub+'_normalized_m256.npz')
        models[cat].append(m)
        idfs[cat].append(idf)

cell = 500
for i in range(15):
    sub = subjects[i]
    for ratio in range(0, 100, 10):
        val = pd.read_csv(path_fixed+sub+'_norm/fixed_'+str(ratio)+'_'+str(100-ratio)+'_'+str(cell)+'_1000_binomial.txt', index_col = 0, nrows = 200)
        X_val, y_val = splitData(val[keep_gene])
        celltypes = y_val.columns

        pred = get_prediction(X_val.values, models, idfs, i, diffs)
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