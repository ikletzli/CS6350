import os
import sys
import math
#import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import minimize
import torch
from torch import nn
from helper import *

def sign(vec):
    signed = np.vectorize(lambda val: -1 if val < 0 else 1)(vec)
    return signed

def svm_prediction(x, y, w):
    predictions = sign(x.dot(w))
    err = 1 - (np.sum(predictions == y) / len(y))
    return err

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
            y = example[15]
            this_x = example[0:15]

            _, cache, L = forward_pass(this_x, y, hidden_size_1, hidden_size_2, w_h_1, w_h_2, w_o)
            dwo, dw_h_2, dw_h_1 = backward_pass(cache)

            w_o = w_o - gamma_t * dwo
            w_h_2 = w_h_2 - gamma_t * dw_h_2
            w_h_1 = w_h_1 - gamma_t * dw_h_1

        y_pred, _, L = forward_pass(x[:,0:15], x[:,15], hidden_size_1, hidden_size_2, w_h_1, w_h_2, w_o)
        #print(x.shape[0])
        print(f"{T+1}:", np.sum(L) / x.shape[0])
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

def evaluate_svm():
    attributes = read_attributes()
    attribute_names = list(attributes.keys())
    train_data = read_examples("income2023f/train_final.csv", attribute_names)
    test_data = read_examples("income2023f/test_final.csv", attribute_names)

    train = convert_to_numpy(train_data, attributes)
    np.random.shuffle(train)
    validation = train[0:len(train)//10,:]
    train = train[len(train)//10:,:]
    test = convert_to_numpy(test_data, attributes)

    #train_x = train[:,0:15]
    # train_y = train[:,15]

    # validation_x = train[:,0:15]
    # validation_y = train[:,15]
    
    # test_x = test[:,0:15]
    # test_y = test[:,15]

    print("Parameters initialized from standard Gaussian distribution:")
    for width in [10]:
        l0 = 5e-1
        a = 5e-1
        #a = 50

        w_h_1 = np.random.normal(size=(15, width))
        w_h_2 = np.random.normal(size=(width + 1, width))
        w_o = np.random.normal(size=(width + 1,))
        w_h_1, w_h_2, w_o = nn_gradient_descent(train, num_epochs=10, l0=l0, a=a, hidden_size_1=width, hidden_size_2=width, w_h_1=w_h_1, w_h_2=w_h_2, w_o=w_o)
        y_pred, _, L = forward_pass(train[:,0:15], train[:,15], width, width, w_h_1, w_h_2, w_o)
        train_err = nn_prediction(train[:,0:15], train[:,15], y_pred)

        y_pred, _, L = forward_pass(validation[:,0:15], validation[:,15], width, width, w_h_1, w_h_2, w_o)
        validation_err = nn_prediction(validation[:,0:15], validation[:,15], y_pred)

        print("\tWidth:", width, "Train Error:", train_err, "Validation Error:", validation_err)

    print("\nParameters initialized to all zeros:")
    for width in [5,10,25,50,100]:
        l0 = 5e-2
        a = 50
        if width in [10]:
            l0 = 5e-2
            a = 10
        if width in [25]:
            l0 = 5e-1
            a = 10
        if width in [50]:
            l0 = 3e-1
            a = 10
        if width in [100]:
            l0 = 5e-2
            a = 0.4

        w_h_1 = np.zeros((15, width))
        w_h_2 = np.zeros((width + 1, width))
        w_o = np.zeros((width + 1,))
        w_h_1, w_h_2, w_o = nn_gradient_descent(x_train, num_epochs=100, l0=l0, a=a, hidden_size_1=width, hidden_size_2=width, w_h_1=w_h_1, w_h_2=w_h_2, w_o=w_o)
        y_pred, _, L = forward_pass(x_train[:,0:15], x_train[:,15], width, width, w_h_1, w_h_2, w_o)
        train_err = nn_prediction(x_train[:,0:15], x_train[:,15], y_pred)

        y_pred, _, L = forward_pass(x_test[:,0:15], x_test[:,15], width, width, w_h_1, w_h_2, w_o)
        test_err = nn_prediction(x_test[:,0:15], x_test[:,15], y_pred)

        print("\tWidth:", width, "Train Error:", train_err, "Test Error:", test_err)

def train_loop(x_train, model, loss_fn, optimizer):
    size = len(x_train)
    all_X = x_train[:,0:15]
    all_y = x_train[:,15]
    model.train()

    for i in range(size):
        X = all_X[i]
        y = all_y[i]
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_loop(x_test, model, loss_fn):
    model.eval()
    size = len(x_test)
    test_loss, correct = 0, 0
    all_X = x_test[:,0:15]
    all_y = x_test[:,15]

    with torch.no_grad():
        for i in range(size):
            X = all_X[i]
            y = all_y[i]
            pred = model(X)
            correct += np.sum(sign(pred.item()) == y.item())
            test_loss += loss_fn(pred, y).item()

    correct /= size
    test_loss /= size
    err = 1 - correct
    return err

def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss

def init_xavier(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

def init_he(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)

def pytorch_training():
    train_data = read_examples("bank-note/train.csv")
    test_data = read_examples("bank-note/test.csv")

    x_train = np.zeros((6, len(train_data)))
    x_train[0,:] = np.ones(len(train_data))

    counter = 0
    for example in train_data:
        x_train[1:, counter] = example
        counter += 1

    x_train = x_train.T

    x_test = np.zeros((6, len(test_data)))
    x_test[0,:] = np.ones(len(test_data))

    counter = 0
    for example in test_data:
        x_test[1:, counter] = example
        counter += 1

    x_test = x_test.T

    print("Tanh activation and xavier initialization:")
    for width in [3,5,9]:
        for depth in [5,10,25,50,100]:
            model = nn.Sequential(
                nn.Linear(15,width),
                nn.Tanh(),
                nn.Linear(width,depth),
                nn.Tanh(),
                nn.Linear(depth,1),
            )
            model.apply(init_xavier)

            loss_fn = my_loss
            optimizer = torch.optim.Adam(model.parameters())

            epochs = 10
            for t in range(epochs):
                train_loop(torch.from_numpy(x_train).float(), model, loss_fn, optimizer)
            
            train_err = test_loop(torch.from_numpy(x_train).float(), model, loss_fn)
            test_err = test_loop(torch.from_numpy(x_test).float(), model, loss_fn)
            print("\tWidth:", width, "Depth:", depth, "Train Error:", train_err, "Test Error:", test_err)

    print("\nReLU activation and he initialization:")
    for width in [3,5,9]:
        for depth in [5,10,25,50,100]:
            model = nn.Sequential(
                nn.Linear(15,width),
                nn.ReLU(),
                nn.Linear(width,depth),
                nn.ReLU(),
                nn.Linear(depth,1),
            )
            model.apply(init_he)

            loss_fn = my_loss
            optimizer = torch.optim.Adam(model.parameters())

            epochs = 10
            for t in range(epochs):
                train_loop(torch.from_numpy(x_train).float(), model, loss_fn, optimizer)
            
            train_err = test_loop(torch.from_numpy(x_train).float(), model, loss_fn)
            test_err = test_loop(torch.from_numpy(x_test).float(), model, loss_fn)
            print("\tWidth:", width, "Depth:", depth, "Train Error:", train_err, "Test Error:", test_err)

def main():
    evaluate_svm()
    pytorch_training()

if __name__ == "__main__":
    main()