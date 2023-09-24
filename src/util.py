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
    idf = np.log(1.0 * X.shape[1] / (np.sum(X,axis=1)+1)+1)
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
    
def alterprocess(X, normalization='FPKM', scaler=None, length_file=None):
    # X (samples, genes)
    if normalization=='FPKM':
        X = counts2FPKM(X, length_file=length_file)
    elif normalization=='TPM':
        X = counts2TPM(X, length_file=length_file)
    elif normalization=='tfidf':
        X, idf_diag = preprocess(X.values)
        return X, idf_diag
    elif normalization=='cpm':
        X = counts2CPM(X)
    X = scale(X, scaler=scaler)  
    idf_diag = None  
    return X, idf_diag

def scale(X, scaler='mms'):
    # X (samples, genes)
    X = np.log(X+1)
    if scaler=='mms':
        processed_X =  MinMaxScaler().fit_transform(X.T).T
    elif scaler=='ss':
        processed_X =  StandardScaler().fit_transform(X.T).T

    return processed_X
    
def splitData(X, binomial=False):
    if binomial:
        x = X.iloc[:, 3:]
        y = pd.DataFrame([X['mal_ratio'], 1-X['mal_ratio']],  index=['malignant', 'normal']).T
    else:  
        x = X.iloc[:, 2:]
        frac = X.iloc[:, :2]
        y = frac.divide(frac.sum(axis=1), axis=0)
    return x, y

def counts2FPKM(counts, length_file=None, genes=None):
    #counts: (samples, genes)
    #length_file: gene length file
    #genes: gene list if counts is a numpy array
    #return: (samples, genes)
    # ref http://luisvalesilva.com/datasimple/rna-seq_units.html
    genefile = pd.read_csv(length_file, sep=',')
    genefile['Length'] = genefile['Transcript end (bp)'] - genefile['Transcript start (bp)']
    genelen = genefile[['Gene name', 'Length']]
    genelen = genelen.groupby('Gene name').max()
    # intersection
    inter = counts.columns.intersection(genelen.index)
    samplename = counts.index

    counts = counts[inter].values
    genelen = genelen.loc[inter].T.values
    totalreads = counts.sum(axis=1)
    counts = counts * 1e9 / (genelen * totalreads.reshape(-1, 1))
    counts = pd.DataFrame(counts, columns=inter, index=samplename)
    return counts

def FPKM2TPM(counts):
    # fpkm: (samples, genes)
    # return: (samples, genes)
    fpkm = counts.values
    totalreads = fpkm.sum(axis=1)
    tpm = fpkm * 1e6 / totalreads.reshape(-1, 1)
    tpm = pd.DataFrame(tpm, columns=counts.columns, index=counts.index)
    return tpm

def counts2CPM(counts):
    #counts: (samples, genes)
    values = counts.values
    totalreads = values.sum(axis=1)
    cpm = values * 1e6 / totalreads.reshape(-1, 1)
    counts = pd.DataFrame(cpm, columns=counts.columns, index=counts.index)
    return counts

def counts2TPM(counts, length_file):
    #counts: (samples, genes)
    #return: (samples, genes)
    fpkm = counts2FPKM(counts, length_file)
    tpm = FPKM2TPM(fpkm)
    return tpm

def ccc(y_true, y_pred):
    # Concordance correlation coefficient
    # https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    rho = np.corrcoef(y_true, y_pred)[0][1]
    ccc = 2*rho*sd_true*sd_pred / (var_true+var_pred+(mean_true-mean_pred)**2)
    return ccc

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
