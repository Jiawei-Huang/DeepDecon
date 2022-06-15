from train_model import *
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt



test_genes = pd.read_csv('./aml_subject_data/common_gene.txt', index_col=0)
keep_gene = ['malignant', 'normal']+list(test_genes['gene'].values)
differences = []
categories = []
for i in range(10):
    if i < 5:
        cat = 'less_'+str((i+1)*10)
    else:
        cat = 'greater_'+str(i*10)
    
    path = './aml_simulated_bulk_data/AML_'+cat+'/'
    categories.append(cat)
    subjects = ['AML328-D29', 'AML1012-D0', 'AML556-D0', 'AML328-D171', 
                'AML210A-D0', 'AML419A-D0', 'AML328-D0', 'AML707B-D0',
                'AML916-D0', 'AML328-D113', 'AML329-D0', 'AML420B-D0',
                'AML329-D20', 'AML921A-D0', 'AML475-D0'
            ]

    diffs = []

    pred_path = path+'prediction/'
    if not os.path.exists(pred_path):
        os.makedirs(pred_path) 
    for sub in subjects:
        tmp = pd.read_csv(path+sub+'_bulk_nor_500_200.txt', index_col=0)
        model = load_model(path+'models/'+sub+'_deepdecon_tf_idf_normalized_m256.h5', custom_objects={'rmse':rmse})
        idf = sp.load_npz(path+'idfs/'+sub+'_normalized_m256.npz')

        val = tmp[keep_gene]
        X_val, y_val = splitData(val)
        
        nor_X_val_scale, _ = preprocess(X_val.values, idf)
        nor_y_val_scale = y_val.values
        pred = model.predict(nor_X_val_scale)
        pd.DataFrame(pred, columns=['malignant', 'normal']).to_csv(pred_path+sub+'_deepdecon_tf_idf_m256_predictions.txt')

        metric = np.sqrt(np.mean((nor_y_val_scale[:,0]-pred[:, 0])**2))
        print(subjects[i], 'RMSE: %.4f'%metric)

        diff = np.abs(nor_y_val_scale[:,0]-pred[:, 0])
        diffs.append(diff)
    diff_total = np.concatenate(diffs)
    differences.append(diff_total) 

pd.DataFrame(differences, index=categories).T.to_csv('./aml_simulated_bulk_data/prediction_abs_difference.txt')


