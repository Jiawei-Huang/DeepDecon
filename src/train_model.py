import os
import random
import pandas as pd
import numpy as np
import argparse

import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras import metrics
from sklearn.model_selection import train_test_split

from util import splitData, preprocess, alterprocess


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(y_true, y_pred)))

class Net(object):
    def __init__(
        self,
        n_in,
        n_out,
        loss,
        model_name,
        hidden,
        dropout, 
        epochs, 
        activation = 'relu',
        optimizer = 'adam',
        large = False,
        early_stopping = False, 
        early_stopping_validation_fraction = 0.1, 
        early_stopping_patience = 40,
        validation_data_for_early_stopping = None,
        batch_size = 128,
        verbose = 0, 
        ):

        self.n_in = n_in
        self.n_out = n_out
        self.loss = loss
        self.name = model_name
        self.hidden = hidden
        self.drop = dropout
        self.act = activation
        self.epochs = epochs
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.early_stopping_validation_fraction = early_stopping_validation_fraction
        self.early_stopping_patience = early_stopping_patience
        self.validation_data_for_early_stopping = validation_data_for_early_stopping
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.model_fn(verbose=True, large=large)

    def model_fn(self, verbose=False, large=False):
        print('Build Model:', self.name)

        model = Sequential()
        model.add(Dense(self.hidden[0], input_dim=self.n_in, activation=self.act, kernel_regularizer=l2(0.001), name='Dense_1'))
        model.add(Dropout(self.drop[0]))
        model.add(Dense(self.hidden[1], activation=self.act, kernel_regularizer=l2(0.001), name='Dense_2'))
        model.add(Dropout(self.drop[1]))
        model.add(Dense(self.hidden[2], activation=self.act, kernel_regularizer=l2(0.001), name='Dense_3'))
        model.add(Dropout(self.drop[2]))
        if large:
            model.add(Dense(self.hidden[3], activation=self.act, kernel_regularizer=l2(0.001), name='Dense_4'))
            model.add(Dropout(self.drop[3]))
        
#             model.add(Dense(self.hidden[4], activation=self.act, kernel_regularizer=l2(0.001), name='Dense_5'))
#             model.add(Dropout(self.drop[4]))
#             model.add(Dense(self.hidden[5], activation=self.act, kernel_regularizer=l2(0.001), name='Dense_6'))
#             model.add(Dropout(self.drop[5]))
        if self.n_out == 1:
            model.add(Dense(self.n_out, activation='sigmoid', name='output'))
        else:
            model.add(Dense(self.n_out, activation='softmax', name='output'))

        if verbose:
            model.summary()

        self.model = model

    def fit(self, X, y, X_val=None, y_val=None, epochs = None, verbose = None):
        self.model.compile(loss=self.loss, optimizer = self.optimizer, metrics=[rmse, 'mse', metrics.mae])

        if epochs is not None:
            self.epochs = epochs
        if verbose is None:
            verbose = self.verbose
        if X_val is None:
            X_tr, X_val, y_tr, y_val = train_test_split(X, 
                                                        y,
                                                        test_size = self.early_stopping_validation_fraction,
                                                        shuffle=True)
        else:
            X_tr, y_tr = X, y
            
        validation_data = (X_val, y_val)

        if self.early_stopping:
            callbacks = [EarlyStopping(monitor = 'val_loss', patience = self.early_stopping_patience, verbose = 1)]
        else:
            callbacks = None
        
        history = self.model.fit(X_tr, y_tr, batch_size = self.batch_size,
        epochs = self.epochs, validation_data = validation_data, callbacks = callbacks, shuffle = True, verbose = verbose)

        return history


def main(data):

    keep_info = {
        'train_ind':[],
        'test_ind':[],
        'keep_genes':[],
        'model':[],
        'idf_tr':[],
        'idf_val':[]
    }

    architectures = {'hidden':[256, 128, 64, 32], 'dropout1': [0.1, 0.1, 0.1, 0], 'dropout2': [0.2, 0.2, 0.2, 0]}
    test_genes = pd.read_csv('./common_gene.txt', index_col=0)
    keep_gene = ['malignant', 'normal']+list(test_genes['gene'].values)
    for i in range(15):
        train_ind = list(range(i))  + list(range(i+1, len(data)))
        print(train_ind)
        print('Test dataset:', subjects[i])
        train = pd.concat([data[j][keep_gene] for j in train_ind], axis=0, ignore_index=True)
        val = data[i][keep_gene]

        X_tr, y_tr = splitData(train)
        X_val, y_val = splitData(val)


        X_tr1, x_tr1_idf = alterprocess(X_tr, normalization=normalization, scaler=scaler)
        celltypes = y_tr.columns
        nor_X_tr = X_tr1
        nor_y_tr = y_tr.values
        nor_X_val_scale, idf_val = alterprocess(X_val, normalization=normalization, scaler=scaler)
        nor_y_val_scale = y_val.loc[:, celltypes].values

        

        epochs = 500
        opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
        m256 = Net(nor_X_tr.shape[1], 2, loss=rmse, model_name='m256', optimizer=opt, large=True, epochs=epochs,
                    early_stopping=True, hidden=architectures['hidden'], dropout=architectures['dropout'+str(dr)], batch_size=bs)
        h_256 = m256.fit(nor_X_tr, nor_y_tr, X_val=nor_X_val_scale, y_val=nor_y_val_scale, epochs=epochs)
        keep_info['model'].append(m256)
        keep_info['idf_tr'].append(x_tr1_idf)
        keep_info['idf_val'].append(idf_val)
        
        model_path = datapath + 'models/' + normalization + '_' + scaler + '/'
        pred_path = datapath + 'predictions/'+ normalization + '_' + scaler + '/'
        

        if not os.path.exists(model_path):
            os.makedirs(model_path) 

        if not os.path.exists(pred_path):
            os.makedirs(pred_path)

        name = subjects[i]
        m256.model.save(model_path+name+'_'+str(args.lr)+'_'+str(bs)+'_'+ str(dr)+'.h5')
        if idf_val is not None:
            idf_path = datapath + 'idfs/'+ normalization + '_' + scaler + '/'
            if not os.path.exists(idf_path):
                os.makedirs(idf_path)
            sp.save_npz(idf_path+name+'_'+str(args.lr)+'_'+str(bs)+'_'+ str(dr)+'.npz', idf_val)
        
        X_val, y_val = splitData(val[keep_gene])
        nor_X_val_scale, idf_val = alterprocess(X_val, normalization=normalization, scaler=scaler)
        preds = m256.model.predict(nor_X_val_scale)
        pd.DataFrame(preds, columns=['malignant', 'normal']).to_csv(pred_path+name+'_'+str(args.lr)+'_'+str(bs)+'_'+ str(dr)+'_prediction_val.txt')
        print('prediction_val saved at ', pred_path+name+'_'+str(args.lr)+'_'+str(bs)+'_'+ str(dr)+'_prediction_val.txt')

 
if __name__ == "__main__":
    # main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, help="Number of cells to use for each bulk sample.", default=500) # here we use 500 to save time, in the manuscript, it's 3000
    parser.add_argument("--path", type=str, help="Training data directory", default='./aml_simulated_bulk_data/')
    parser.add_argument("--lr", type=int, help="learning rate index k, lr = 10^(-k)", default=4)
    parser.add_argument("--bs", type=int, help="batch size", default=128)
    parser.add_argument("--dr", type=str, help="dropout", default='1')
    parser.add_argument("--start", type=int, help="Fraction start range of generated samples e.g. 0 for [0, 100]", default=0)
    parser.add_argument("--end", type=int, help="Fraction end range of generated samples e.g. 100 for [0, 100]", default=100)
    parser.add_argument("--scaler", type=str, help="Scaler of neural network, MinMaxScaler (mms) or StandardScaler (ss)", default='mms')
    parser.add_argument("--normalization", type=str, help="Normalization methods,TF-IDF, FPKM, CPM or TPM", default='tfidf')
    
    args = parser.parse_args()

    cell = args.cells
    path = args.path
    start = args.start
    end = args.end
    scaler = args.scaler
    normalization = args.normalization
    # print(start, end, scaler, normalization)
    lr = 10**(-args.lr)
    bs = args.bs
    dr = args.dr
    # print(lr, bs, dr)
    subjects = ['AML328-D29', 'AML1012-D0', 'AML556-D0', 'AML328-D171', 
                'AML210A-D0', 'AML419A-D0', 'AML328-D0', 'AML707B-D0',
                'AML916-D0', 'AML328-D113', 'AML329-D0', 'AML420B-D0',
                'AML329-D20', 'AML921A-D0', 'AML475-D0'
            ]

    data_path = path + "sample_" + str(cell) + "/range_" + str(start) + "_" + str(end) + "/"
    data = []

    for sub in subjects:
        tmp = pd.read_csv(data_path+sub+'_bulk_nor_500_200.txt', index_col=0)
        data.append(tmp)
        print(sub, ' loaded')
    main(data)
