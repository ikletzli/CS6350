import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from helper import *
from scipy.optimize import minimize

feature_size = 15

def stochastic_gradient_descent(x, y, C):
    r = 5e-9
    w = np.zeros((feature_size))

    costs = []
    iterations = 1
    while iterations < 1000:
        index = random.randint(0, y.size - 1)
        new_x = x[index,:]
        new_y = y[index]
        #cost = np.sum(np.log(1+np.exp(-y*x.dot(w)))) + C * w.dot(w)
        # print("Reg:", C * w.dot(w))
        # print("Problem:", -y*x.dot(w))
        # print("Problem:", -y*x.dot(w))
        # print("Other:", 1+np.exp(-y*x.dot(w)))
        cost = np.sum(np.log(1+np.exp(-y*x.dot(w)))) + C * w.dot(w)
        print(cost)
        costs.append(cost)
        #print(cost)

        # grad = np.zeros_like(w)
        # for i in range(w.size):
        #     grad[i] = -(new_y - w.dot(new_x)) * new_x[i]

        grad = (1/(1+np.exp(-new_y*new_x.dot(w))) * np.exp(-new_y*new_x.dot(w)) * -new_y*new_x + 2 * C * w) / y.size
        
        
        
        
        # new_w = w - r*grad
        # print(grad)
        # print(new_w)
        # break




        #print("Grad:", grad)
        
        new_w = w - r*grad
        # if np.linalg.norm(new_w - w) < 1e-6:
        #     w = new_w
        #     break

        w = new_w
        iterations += 1

    return costs, w, r

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

def svm_dual_form(x, y, C):
    fun = lambda w: np.sum(np.log(1+np.exp(-y*x.dot(w)))) + C * w.dot(w)

    guess = []

    for i in range(x.shape[1]):
        guess.append(1)
    
    #res = minimize(fun, guess, method='SLSQP')
    res = minimize(fun, guess)

    w_star = res.x
    print("Done")
    return w_star

def sign(vec):
    signed = np.vectorize(lambda val: -1 if val < 0.5 else 1)(vec)
    return signed

def lms():
    attributes = read_attributes()
    attribute_names = list(attributes.keys())
    train_data = read_examples("income2023f/train_final.csv", attribute_names)
    test_data = read_examples("income2023f/test_final.csv", attribute_names)

    train = convert_to_numpy_small(train_data, attributes)
    np.random.shuffle(train)
    validation = train[0:len(train)//10,:]
    train = train[len(train)//10:,:]
    test = convert_to_numpy_small(test_data, attributes)

    train_x = train[0:20000,0:feature_size]
    train_y = train[0:20000,feature_size]

    validation_x = train[20000:,0:feature_size]
    validation_y = train[20000:,feature_size]
    
    test_x = test[:,0:feature_size]
    test_y = test[:,feature_size]

    #w_star = svm_dual_form(train_x, train_y,0.1)
    costs, w, r = stochastic_gradient_descent(train_x, train_y,1e-2)
    predictions = sign(sigmoid(validation_x.dot(w)))
    print(1 - (np.sum(predictions == validation_y) / len(validation_y)))

    labels = sigmoid(test_x.dot(w))

    save_csv(labels, "logistic")

    # predictions = sign(sigmoid(validation_x.dot(w_star)))

    # print(1 - (np.sum(predictions == validation_y) / len(validation_y)))


    
    #print(predictions)

def main():
    lms()

if __name__ == "__main__":
    main()