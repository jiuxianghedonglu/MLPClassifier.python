# -*- coding: utf-8 -*-
"""
  @Author: zzn
  @Date: 2019-01-02 14:36:33
  @Last Modified by:   zzn
  @Last Modified time: 2019-01-02 14:36:33
"""
import numpy as np
import pickle


def sigmoid(x):
    return 1/(1+np.exp(-x))


def softmax(x):
    exp_x = np.exp(x)
    sum_expx = np.sum(exp_x, axis=-1)
    sum_expx = sum_expx[:, np.newaxis]
    return exp_x/sum_expx


class MLP(object):
    def __init__(self, in_size=10, out_size=3, hidden_sizes=(100,)):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes)+2
        self.parameters = self.initializing_paras()
        self.hidden_activation = sigmoid
        self.out_activation = softmax

    def initializing_paras(self):
        paras = {}
        in_dim = self.in_size
        for i, h_dim in enumerate(self.hidden_sizes):
            out_dim = h_dim
            w_tmp = np.random.standard_normal((in_dim, out_dim))
            b_tmp = np.random.standard_normal(out_dim)
            paras['W_{}'.format(i)] = w_tmp
            paras['b_{}'.format(i)] = b_tmp
            in_dim = out_dim
        out_dim = self.out_size
        w_tmp = np.random.standard_normal((in_dim, out_dim))
        b_tmp = np.random.standard_normal(out_dim)
        paras['W_{}'.format(i+1)] = w_tmp
        paras['b_{}'.format(i+1)] = b_tmp
        return paras

    def save_weights(self, full_path):
        with open(full_path, 'wb') as f:
            pickle.dump(self.parameters, f)

    def load_weights(self, full_path):
        with open(full_path, 'rb') as f:
            self.parameters = pickle.load(f)

    def predict(self, x, batch_size=32):
        n, d = x.shape
        output = np.zeros((n, self.out_size))
        for i in range(0, n, batch_size):
            batch_x = x[i:i+batch_size]
            batch_y, _ = self.forward_batch(batch_x)
            output[i:i+batch_size] = batch_y
        return output

    def forward_batch(self, batch_x):
        batch_size = batch_x.shape[0]
        out_list = []
        out_list.append(batch_x)
        for i in range(self.n_layers-1):
            w_tmp = self.parameters['W_{}'.format(i)]
            b_tmp = self.parameters['b_{}'.format(i)]
            batch_x = np.dot(batch_x, w_tmp)+b_tmp
            if i != self.n_layers-2:
                batch_x = self.hidden_activation(batch_x)
                out_list.append(batch_x)
            else:
                batch_x = self.out_activation(batch_x)
                out_list.append(batch_x)
        batch_y = batch_x
        return batch_y, out_list

    def backward_batch(self, batch_y_true, output_list):
        param_grads = {}
        batch_size = batch_y_true.shape[0]
        for i in range(self.n_layers-1, 0, -1):
            output = output_list[i]
            h = output_list[i-1]
            if i == self.n_layers-1:
                dL_ds = output-batch_y_true
                tmp = dL_ds
            else:
                dL_ds = output*(1-output)*np.dot(tmp,
                                                 self.parameters['W_{}'.format(i)].T)
                tmp = dL_ds
            ds_dw = h
            dL_dw = np.dot(ds_dw.T, dL_ds)
            dL_db = dL_ds.sum(axis=0)
            param_grads['W_{}'.format(i-1)] = dL_dw
            param_grads['b_{}'.format(i-1)] = dL_db
        return param_grads

    def update_params_sgd(self, param_grads, lr, v, momentum):
        for i, key in enumerate(param_grads):
            v[key] = momentum*v[key]-lr*param_grads[key]
            self.parameters[key] += v[key]
        return v

    def fit(self, x, y, val_x=None, val_y=None, batch_size=32, epochs=100, lr=1e-2, momentum=0):
        n = x.shape[0]
        indexs = np.arange(x.shape[0])
        for e in range(1, epochs+1):
            np.random.shuffle(indexs)
            x = x[indexs]
            y = y[indexs]
            for i in range(0, n, batch_size):
                batch_x = x[i:i+batch_size]
                batch_y_true = y[i:i+batch_size]
                batch_y_pred, out_list = self.forward_batch(batch_x)
                param_grads = self.backward_batch(batch_y_true, out_list)
                if e == 1 and i == 0:
                    v = {key: 0
                         for i, key in enumerate(param_grads)}
                else:
                    v = self.update_params_sgd(
                        param_grads, lr=lr, v=v, momentum=momentum)
            pred_y = self.predict(x)
            pred_y = np.argmax(pred_y, axis=1)
            true_y = np.argmax(y, axis=1)
            val_pred_y = self.predict(val_x)
            val_pred_y = np.argmax(val_pred_y, axis=1)
            val_true_y = val_y
            print('train acc:{:.4f} \t val acc:{:.4f}'.format(
                np.mean(pred_y == true_y), np.mean(val_pred_y == val_true_y)))
