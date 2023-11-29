import os
import sys
import math
#import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import minimize

# converts the examples into a list of maps that map an example's attributes to its values
def read_examples(file_name):
    attributes = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, file_name)
    examples = []

    with open (file_path, 'r') as f:
        for line in f:
            values = line.strip().split(',')
            example = np.zeros(5)
            for i in range(len(attributes)):
                example[i] = values[i]

                if attributes[i] == 'label':
                    if values[i] == '0':
                        example[i] = -1

            examples.append(example)
    
    return examples

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

def svm_kernel_prediction(x, y, gamma, a_star, C):
    w_star_x = np.zeros((1, x.shape[0]))
    b_star = 0
    count = 0

    for i in range(x.shape[0]):
        w_star_x += a_star[i] * y[i] * kernel(x[i], x, gamma)        
        if (a_star[i] > 0 and a_star[i] < C):
            count = count + 1
            w_t_kernel_x_j = 0
            for j in range(x.shape[0]):
                w_t_kernel_x_j += y[j] * a_star[j] * kernel(x[i], x[j], gamma)

            b_star += y[i] - w_t_kernel_x_j

    b_star = b_star / count

    predictions = sign(w_star_x + b_star)
    err = 1 - (np.sum(predictions == y) / len(y))
    return err

def schedule(l0, t, a):
    return l0 / (1 + (l0/a) * t)

def svm_dual_form(train, C):
    x = train[:,1:5]
    y = train[:,5]
    x_x_y_y = np.outer(y, y) * x.dot(x.T)
    fun = lambda a: 0.5 * np.sum(np.outer(a, a) * x_x_y_y) - np.sum(a)

    bnds = []
    guess = []

    for i in range(x.shape[0]):
        bnds.append((0,C))
        guess.append(0)

    cons = ({'type': 'eq', 'fun': lambda a: a.dot(y)})

    res = minimize(fun, guess, method='SLSQP', bounds=bnds, constraints=cons)

    a_star = res.x
    w_star = x.T.dot(a_star * y)

    count = 0
    b_star = 0
    for i in range(x.shape[0]):
        if (a_star[i] > 0 and a_star[i] < C):
            count = count + 1
            b_star = b_star + y[i] - w_star.dot(x[i])

    b_star = b_star / count

    augmented = np.zeros((5,))
    augmented[0] = b_star
    augmented[1:5] = w_star
    return augmented

def kernel(x, z, gamma):
    if x.ndim == 2:
        return_val = np.zeros((x.shape[0], z.shape[0]))
        for i in range(x.shape[0]):
            x_i = x[i]
            for j in range(z.shape[0]):
                z_j = z[j]
                return_val[i,j] = np.exp(-(np.linalg.norm(x_i - z_j)**2)/gamma)

        return return_val
    
    else:
        if z.ndim == 2:
            return_val = np.zeros((1, z.shape[0]))
            for j in range(z.shape[0]):
                z_j = z[j]
                return_val[0,j] = np.exp(-(np.linalg.norm(x - z_j)**2)/gamma)

            return return_val
        
        else:
            return np.exp(-(np.linalg.norm(x - z)**2)/gamma)

def svm_gaussian_kernel(train, C, gamma):
    x = train[:,1:5]
    y = train[:,5]
    x_x_y_y = np.outer(y, y) * kernel(x, x, gamma)
    fun = lambda a: 0.5 * np.sum(np.outer(a, a) * x_x_y_y) - np.sum(a)

    bnds = []
    guess = []

    for i in range(x.shape[0]):
        bnds.append((0,C))
        guess.append(0)

    cons = ({'type': 'eq', 'fun': lambda a: a.dot(y)})

    res = minimize(fun, guess, method='SLSQP', bounds=bnds, constraints=cons)

    return res.x

def nn_gradient_descent(x, num_epochs, l0, a, hidden_size_1, hidden_size_2, w_h_1, w_h_2, w_o):    
    gamma_t = l0

    for T in range(num_epochs):
        np.random.shuffle(x)
        for example in x:
            y = example[5]
            this_x = example[0:5]

            _, cache, L = forward_pass(this_x, y, hidden_size_1, hidden_size_2, w_h_1, w_h_2, w_o)
            dwo, dw_h_2, dw_h_1 = backward_pass(cache)

            w_o = w_o - gamma_t * dwo
            w_h_2 = w_h_2 - gamma_t * dw_h_2
            w_h_1 = w_h_1 - gamma_t * dw_h_1

        y_pred, _, L = forward_pass(x[:,0:5], x[:,5], hidden_size_1, hidden_size_2, w_h_1, w_h_2, w_o)
        #print(x.shape[0])
        #print(np.sum(L) / x.shape[0])
        gamma_t = schedule(l0, T+1, a)

    return w_h_1, w_h_2, w_o

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

    print("Parameters initialized from standard Gaussian distribution:")
    for width in [5,10,25,50,100]:
        l0 = 5e-1
        a = 50
        if width in [25]:
            l0 = 5e-2
            a = 10
        if width in [50]:
            l0 = 5e-3
            a = 10
        if width in [100]:
            l0 = 5e-4
            a = 10

        w_h_1 = np.random.normal(size=(5, width))
        w_h_2 = np.random.normal(size=(width + 1, width))
        w_o = np.random.normal(size=(width + 1,))
        w_h_1, w_h_2, w_o = nn_gradient_descent(x_train, num_epochs=100, l0=l0, a=a, hidden_size_1=width, hidden_size_2=width, w_h_1=w_h_1, w_h_2=w_h_2, w_o=w_o)
        y_pred, _, L = forward_pass(x_train[:,0:5], x_train[:,5], width, width, w_h_1, w_h_2, w_o)
        train_err = nn_prediction(x_train[:,0:5], x_train[:,5], y_pred)

        y_pred, _, L = forward_pass(x_test[:,0:5], x_test[:,5], width, width, w_h_1, w_h_2, w_o)
        test_err = nn_prediction(x_test[:,0:5], x_test[:,5], y_pred)

        print("\tWidth:", width, "Train Error:", train_err, "Test Error:", test_err)

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

        w_h_1 = np.zeros((5, width))
        w_h_2 = np.zeros((width + 1, width))
        w_o = np.zeros((width + 1,))
        w_h_1, w_h_2, w_o = nn_gradient_descent(x_train, num_epochs=100, l0=l0, a=a, hidden_size_1=width, hidden_size_2=width, w_h_1=w_h_1, w_h_2=w_h_2, w_o=w_o)
        y_pred, _, L = forward_pass(x_train[:,0:5], x_train[:,5], width, width, w_h_1, w_h_2, w_o)
        train_err = nn_prediction(x_train[:,0:5], x_train[:,5], y_pred)

        y_pred, _, L = forward_pass(x_test[:,0:5], x_test[:,5], width, width, w_h_1, w_h_2, w_o)
        test_err = nn_prediction(x_test[:,0:5], x_test[:,5], y_pred)

        print("\tWidth:", width, "Train Error:", train_err, "Test Error:", test_err)

def main():
    evaluate_svm()

if __name__ == "__main__":
    main()