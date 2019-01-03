# -*- coding: utf-8 -*-
"""
  @Author: zzn 
  @Date: 2019-01-03 19:47:18 
  @Last Modified by:   zzn 
  @Last Modified time: 2019-01-03 19:47:18 
"""
from mlp import MLP
import pandas as pd
import numpy as np

if __name__ == '__main__':
    tr_data = pd.read_csv('mnist_train.csv', header=None)
    val_data = pd.read_csv('mnist_test.csv', header=None)
    tr_data = tr_data.values
    val_data = val_data.values
    tr_x = tr_data[:, 1:]
    tr_y = tr_data[:, 0]
    val_x = val_data[:, 1:]
    val_y = val_data[:, 0]
    print(tr_x.shape, tr_y.shape, val_x.shape, val_y.shape)
    tr_x = tr_x / 255
    val_x = val_x / 255
    tr_y_onehot = np.zeros((tr_y.shape[0], 10))
    for i, y in enumerate(tr_y):
        tr_y_onehot[i][y] = 1
    mlp = MLP(in_size=tr_x.shape[1], out_size=10, hidden_sizes=(100,))
    lr = 1e-2
    momentum = 0
    epochs = 20
    mlp.fit(tr_x, tr_y_onehot, val_x=val_x, val_y=val_y,
            batch_size=256, epochs=epochs, lr=lr, momentum=momentum)
    mlp.save_weights('mlp_weights.model')
    mlp.load_weights('mlp_weights.model')
    val_pred = mlp.predict(val_x)
    val_classes = np.argmax(val_pred, axis=1)
    print('validation accuracy:{}'.format(np.sum(val_classes == val_y)))
