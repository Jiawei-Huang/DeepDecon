import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def tf_trans(X, scaler=None):
    # X (genes, samples)
    if scaler is None:
        return 1.0 * X / np.tile(np.sum(X,axis=0), (X.shape[0],1))
    else:
        return 1.0 * X /scaler.reshape(1, X.shape[1])

def tf_idf(X):
    #TF-IDF transformation
    idf = np.log(1.0 * X.shape[1] / np.sum(X,axis=1)+1)
    idf_diag = sp.diags(list(idf), offsets=0, shape=(X.shape[0], X.shape[0]), format="csr")
    X = idf_diag * tf_trans(X)
    
    return X, idf_diag

def preprocess(X, idf_diag=None, bulk_num = None):
    # X (samples, genes)
    if bulk_num is not None:
        X = X/bulk_num
    if idf_diag is None:
        X, idf_diag = tf_idf(X.T)
    else:
        X = idf_diag*tf_trans(X.T)
    X[np.isnan(X)] = 0
    if len(X) > 1:
        X = MinMaxScaler().fit_transform(X).T
    return X, idf_diag

def splitData(X, binomial=False):
    if binomial:
        x = X.iloc[:, 3:]
        y = pd.DataFrame([X['mal_ratio'], 1-X['mal_ratio']],  index=['malignant', 'normal']).T
    else:  
        x = X.iloc[:, 2:]
        frac = X.iloc[:, :2]
        y = frac.divide(frac.sum(axis=1), axis=0)
    return x, y

def scatter_plot(x, y, title, xlabel='Truth', ylabel='Prediction', path=None):
    plt.figure(figsize=(8, 8))
    plt.scatter(x['malignant'], y['malignant'])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if path is not None:
        plt.savefig(path)
    plt.show()
    plt.close()