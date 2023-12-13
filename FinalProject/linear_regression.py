import os
import sys
import math
import numpy as np
import random
from helper import *

feature_size = 15

def batch_gradient_descent(x, y):
    r = 0.01
    w = np.zeros_like(x[0,:])

    costs = []
    iterations = 1
    while True:
        cost = 0.5 * np.sum((y - w.dot(x))**2)
        costs.append(cost)

        grad = np.zeros_like(w)
        for i in range(w.size):
            grad[i] = -np.sum((y - w.dot(x)) * x[i,:])

        new_w = w - r*grad
        print(np.linalg.norm(new_w - w))
        if np.linalg.norm(new_w - w) < 1e-6:
            w = new_w
            break

        w = new_w
        iterations += 1

    return costs, w, r

def stochastic_gradient_descent(x, y):
    r = 1e-10
    w = np.zeros_like(x.T[:,0])

    costs = []
    iterations = 1
    while True:
        index = random.randint(0, y.size - 1)
        new_x = x.T[:,index]
        new_y = y[index]
        cost = 0.5 * np.sum((y - w.dot(x.T))**2)
        costs.append(cost)
        print(cost)

        grad = np.zeros_like(w)
        for i in range(w.size):
            grad[i] = -(new_y - w.dot(new_x)) * new_x[i]
        
        new_w = w - r*grad
        if np.linalg.norm(new_w - w) < 1e-6:
            w = new_w
            break

        w = new_w
        iterations += 1

    return costs, w, r

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

    train_x = train[:,0:feature_size]
    train_y = train[:,feature_size]

    validation_x = train[:,0:feature_size]
    validation_y = train[:,feature_size]
    
    test_x = test[:,0:feature_size]
    test_y = test[:,feature_size]

    #b_costs, b_w, b_r = batch_gradient_descent(train_x, train_y)
    s_costs, s_w, s_r = stochastic_gradient_descent(train_x, train_y)

    #analytical_optimal_w = np.linalg.inv(train_x.dot(train_x.T)).dot(train_x).dot(train_y)

    #print(f"Analytical optimal w: {analytical_optimal_w}\n")
    #print(f"W from batch gradient descent using a learning rate of {b_r}: {b_w}\n")
    print(f"W from stochastic gradient descent using a learning rate of {s_r}: {s_w}\n")

    # x = np.zeros((8, len(test_data)))
    # x[0,:] = np.ones(len(test_data))
    # y = np.zeros(len(test_data))

    # counter = 0
    # for example in test_data:
    #     x[1:, counter] = example[0:7]
    #     y[counter] = example[7]
    #     counter += 1

    # b_test_cost = 0.5 * np.sum((y - b_w.dot(x))**2)
    # s_test_cost = 0.5 * np.sum((y - s_w.dot(x))**2)

def main():
    lms()

if __name__ == "__main__":
    main()