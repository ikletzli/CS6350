import os
import sys
import math
#import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import minimize
from helper import *

feature_size = 15

def sign(vec):
    signed = np.vectorize(lambda val: -1 if val < 0 else 1)(vec)
    return signed

def svm_prediction(x, y, w):
    predictions = sign(x.dot(w))
    err = 1 - (np.sum(predictions == y) / len(y))
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
    x = train[:,1:feature_size]
    y = train[:,feature_size]
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

    augmented = np.zeros((feature_size,))
    augmented[0] = b_star
    augmented[1:feature_size] = w_star
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
    x = train[:,1:feature_size]
    y = train[:,feature_size]
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

def svm_gradient_descent(x, num_epochs, C, l0, a):
    w = np.zeros((feature_size,))
    
    gamma_t = l0

    for T in range(num_epochs):
        np.random.shuffle(x)
        for example in x:
            y = example[5]
            this_x = example[0:feature_size]
            grad_w = np.zeros((feature_size,))
            grad_w[1:feature_size] = w[1:feature_size]
            if (y * w.dot(this_x) <= 1):
                w = w - gamma_t * grad_w + gamma_t * C * x.shape[0] * y * this_x
            else:
                w = w - gamma_t * grad_w

        gamma_t = schedule(l0, T+1, a)

    return w

def evaluate_svm():
    attributes = read_attributes()
    attribute_names = list(attributes.keys())
    train_data = read_examples("income2023f/train_final.csv", attribute_names)
    test_data = read_examples("income2023f/test_final.csv", attribute_names)

    train = convert_to_numpy_small(train_data, attributes)
    np.random.shuffle(train)
    validation = train[0:len(train)//10,:]
    train = train[len(train)//10:,:]
    test = convert_to_numpy_small(test_data, attributes)
    
    #train_x = train[:,0:feature_size]
    # train_y = train[:,feature_size]

    # validation_x = train[:,0:feature_size]
    # validation_y = train[:,feature_size]
    
    # test_x = test[:,0:feature_size]
    # test_y = test[:,feature_size]

    # values that lead to convergence. l0 and a being the same leads to the schedule in part b
    l0 = [3e-6, 5e-5]
    a = [6e-3, 5e-5]

    print("Problem 2\n")

    for i in range(2):
        if i == 0:
            print("Values for the schedule in part a using l0 =", l0[i], "and a =", a[i])
        else:
            print("Values for the schedule in part b using l0 =", l0[i])
        for C in [100/873, 500/873, 700/873]:
            w = svm_gradient_descent(train,num_epochs=100, C=C, l0=l0[i], a=a[i])
            test_err = svm_prediction(validation[:,0:feature_size], validation[:,feature_size], w)
            train_err = svm_prediction(train[:,0:feature_size], train[:,feature_size], w)

            print_c = None
            if C == 100/873:
                print_c = "100/873"
            elif C == 500/873:
                print_c = "500/873"
            else:
                print_c = "700/873"

            print("C:", print_c, "Test Error:", test_err, "Train Error:", train_err, "w:", w)

        print()

    print("Problem 3 part a\n")

    for C in [100/873, 500/873, 700/873]:
        w = svm_dual_form(train, C=C)
        test_err = svm_prediction(validation[:,0:feature_size], validation[:,feature_size], w)
        train_err = svm_prediction(train[:,0:feature_size], train[:,feature_size], w)

        print_c = None
        if C == 100/873:
            print_c = "100/873"
        elif C == 500/873:
            print_c = "500/873"
        else:
            print_c = "700/873"

        print("C:", print_c, "Test Error:", test_err, "Train Error:", train_err, "w:", w)

    prev_vectors = []

    print("Problem 3 parts b and c\n")

    for C in [100/873, 500/873, 700/873]:
        for gamma in [0.1, 0.5, 1, 5, 100]:
            a_star = svm_gaussian_kernel(train, C=C, gamma=gamma)
            test_err = svm_kernel_prediction(validation[:,0:feature_size], validation[:,feature_size], gamma, a_star, C)
            train_err = svm_kernel_prediction(train[:,0:feature_size], train[:,feature_size], gamma, a_star, C)
            support_vectors = a_star != 0
            num_support_vectors = np.sum(support_vectors)

            print_c = None
            num_overlap = None
            if C == 100/873:
                print_c = "100/873"
            elif C == 500/873:
                if len(prev_vectors) != 0:
                    num_overlap = np.sum(prev_vectors[len(prev_vectors) - 1] * support_vectors)

                prev_vectors.append(support_vectors)
                print_c = "500/873"
            else:
                print_c = "700/873"

            if C == 500/873:
                if (num_overlap == None):
                    print("gamma:", gamma, "C:", print_c, "Test Error:", test_err, "Train Error:", train_err, "Number of Support Vectors:" , num_support_vectors, "Overlap with previous: N/A")
                else:
                    print("gamma:", gamma, "C:", print_c, "Test Error:", test_err, "Train Error:", train_err, "Number of Support Vectors:" , num_support_vectors, "Overlap with previous:", num_overlap)
            else:
                print("gamma:", gamma, "C:", print_c, "Test Error:", test_err, "Train Error:", train_err, "Number of Support Vectors:", num_support_vectors)

def main():
    evaluate_svm()

if __name__ == "__main__":
    main()