import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import minimize
import torch
from torch import nn
from helper import *

feature_size = 15

def sign(vec):
    signed = np.vectorize(lambda val: -1 if val < 0 else 1)(vec)
    return signed

def nn_prediction(x, y, y_pred):
    err = 1 - (np.sum(sign(y_pred) == y) / len(y))
    return err

def schedule(l0, t, a):
    return l0 / (1 + (l0/a) * t)

def nn_gradient_descent(x, num_epochs, l0, a, hidden_size_1, hidden_size_2, w_h_1, w_h_2, w_o):    
    gamma_t = l0

    for T in range(num_epochs):
        np.random.shuffle(x)
        for example in x:
            y = example[feature_size]
            this_x = example[0:feature_size]

            _, cache, L = forward_pass(this_x, y, hidden_size_1, hidden_size_2, w_h_1, w_h_2, w_o)
            dwo, dw_h_2, dw_h_1 = backward_pass(cache)

            w_o = w_o - gamma_t * dwo
            w_h_2 = w_h_2 - gamma_t * dw_h_2
            w_h_1 = w_h_1 - gamma_t * dw_h_1

        y_pred, _, L = forward_pass(x[:,0:feature_size], x[:,feature_size], hidden_size_1, hidden_size_2, w_h_1, w_h_2, w_o)
        #print(x.shape[0])
        #print(f"{T+1}:", np.sum(L) / x.shape[0])
        gamma_t = schedule(l0, T+1, a)

    return w_h_1, w_h_2, w_o

# from cs231n from stanford
def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

def forward_pass(x, y, hidden_size_1, hidden_size_2, w_h_1, w_h_2, w_o):
    z_1 = np.ones((hidden_size_1 + 1,))
    if (x.ndim > 1):
        z_1 = np.ones((hidden_size_1 + 1,x.shape[0]))

    z_2 = np.ones((hidden_size_2 + 1,))
    if (x.ndim > 1):
        z_2 = np.ones((hidden_size_2 + 1,x.shape[0]))

    s_1 = x.dot(w_h_1)
    z_1[1:hidden_size_1+1] = sigmoid(s_1).T
    s_2 = z_1.T.dot(w_h_2)
    z_2[1:hidden_size_2+1] = sigmoid(s_2).T
    y_pred = z_2.T.dot(w_o)
    L = 0.5 * (y_pred - y) ** 2

    cache = (y_pred, y, z_2, w_o, hidden_size_2, s_2, z_1, w_h_2, hidden_size_1, s_1, x, w_h_1)

    return y_pred, cache, L

def backward_pass(cache):
    y_pred, y, z_2, w_o, hidden_size_2, s_2, z_1, w_h_2, hidden_size_1, s_1, x, w_h_1 = cache
    dy = y_pred - y
    dwo = z_2 * dy
    dz2 = w_o * dy
    ds2 = dz2[1:hidden_size_2+1] * sigmoid(s_2) * (1 - sigmoid(s_2))
    dw_h_2 = (z_1.reshape( z_1.shape[0], 1)).dot(ds2.reshape((1, ds2.shape[0])))
    dz_1 = w_h_2.dot(ds2)
    ds1 = dz_1[1:hidden_size_1+1].T * sigmoid(s_1) * (1 - sigmoid(s_1))
    dw_h_1 = x.reshape(x.shape[0],1).dot(ds1.reshape((1, ds1.shape[0])))
    dx = ds1 * w_h_1

    return dwo, dw_h_2, dw_h_1

def train_with_nn():
    attributes = read_attributes()
    attribute_names = list(attributes.keys())
    train_data = read_examples("income2023f/train_final.csv", attribute_names)
    test_data = read_examples("income2023f/test_final.csv", attribute_names)

    # train on full training set
    train = convert_to_numpy_small(train_data, attributes)
    # np.random.shuffle(train)
    # validation = train[0:len(train)//10,:]
    # train = train[len(train)//10:,:]
    test = convert_to_numpy_small(test_data, attributes)

    print("Parameters initialized from standard Gaussian distribution:")
    for width in [80]:
        for l0 in [5e-4]:
            for a in [5e-4]:
                w_h_1 = np.random.normal(size=(feature_size, width))
                w_h_2 = np.random.normal(size=(width + 1, width))
                w_o = np.random.normal(size=(width + 1,))
                w_h_1, w_h_2, w_o = nn_gradient_descent(train, num_epochs=10, l0=l0, a=a, hidden_size_1=width, hidden_size_2=width, w_h_1=w_h_1, w_h_2=w_h_2, w_o=w_o)
                y_pred, _, L = forward_pass(train[:,0:feature_size], train[:,feature_size], width, width, w_h_1, w_h_2, w_o)
                train_err = nn_prediction(train[:,0:feature_size], train[:,feature_size], y_pred)

                # y_pred, _, L = forward_pass(validation[:,0:feature_size], validation[:,feature_size], width, width, w_h_1, w_h_2, w_o)
                # validation_err = nn_prediction(validation[:,0:feature_size], validation[:,feature_size], y_pred)

                y_pred, _, L = forward_pass(test[:,0:feature_size], test[:,feature_size], width, width, w_h_1, w_h_2, w_o)
                save_csv(y_pred, "best_nn")
                #validation_err = nn_prediction(validation[:,0:feature_size], validation[:,feature_size], y_pred)

                #print("Width:", width, "Train Error:", train_err, "Validation Error:", validation_err)

def main():
    train_with_nn()

if __name__ == "__main__":
    main()