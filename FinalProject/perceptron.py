import os
import sys
import math
#import matplotlib.pyplot as plt
import numpy as np
import random
from helper import *

def vanilla_perceptron(x, y, r, num_epochs):
    w = np.zeros_like(x[0])

    for T in range(num_epochs):
        for i in range(len(x)):
            mistake = (y[i] * w.dot(x[i])) <= 0

            if mistake:
                w += r * y[i] * x[i]

    return w

def voted_perceptron(x, y, r, num_epochs):
    w = np.zeros_like(x[0])
    m = 0

    return_vals = [(w,m)]

    for T in range(num_epochs):
        for i in range(len(x)):
            mistake = (y[i] * return_vals[m][0].dot(x[i])) <= 0
            if mistake:
                return_vals.append((return_vals[m][0] + r*y[i]*x[i], 1))
                m += 1
            else:
                return_vals[m] = (return_vals[m][0], return_vals[m][1] + 1)

    return return_vals[1:]

def average_perceptron(x, y, r, num_epochs):
    w = np.zeros_like(x[0])
    a = np.zeros_like(x[0])

    for T in range(num_epochs):
        for i in range(len(x)):
            mistake = (y[i] * w.dot(x[i])) <= 0
            if mistake:
                w += r * y[i] * x[i]
            a += w

    return a

def sign(vec):
    signed = np.vectorize(lambda val: -1 if val < 0 else 1)(vec)
    return signed

def perceptron_prediction(x, y, w):
    predictions = sign(x.dot(w))
    err = 1 - (np.sum(predictions == y) / len(y))
    return err

def voted_perceptron_prediction(x, y, return_vals):
    prediction_sum = np.zeros((len(x)))
    for weight, count in return_vals:
        predictions = sign(x.dot(weight)) * count
        prediction_sum += predictions

    prediction_sum = sign(prediction_sum)
    err = 1 - (np.sum(prediction_sum == y) / len(y))
    return err

def evaluate_perceptron():
    attributes = read_attributes()
    attribute_names = list(attributes.keys())
    train_data = read_examples("income2023f/train_final.csv", attribute_names)
    test_data = read_examples("income2023f/test_final.csv", attribute_names)

    train = convert_to_numpy(train_data, attributes)
    # np.random.shuffle(train)
    # validation = train[0:len(train)//10,:]
    # train = train[len(train)//10:,:]
    test = convert_to_numpy(test_data, attributes)

    train_x = train[:,0:114]
    train_y = train[:,114]

    validation_x = train[:,0:114]
    validation_y = train[:,114]
    
    test_x = test[:,0:114]
    print(test_x[0])
    test_y = test[:,114]

    a = average_perceptron(train_x,train_y,r=0.01,num_epochs=50)
    average_err = perceptron_prediction(validation_x, validation_y, a)

    print(test_x[0])
    print("train", train_x[0])
    predictions = sign(test_x.dot(a))
    save_csv(predictions, "perceptron1")
    print(f"Weight vector for average perceptron: \n\t{a}")
    print(f"Prediction error on test set for average perceptron: \n\t{average_err}")

def main():
    evaluate_perceptron()

if __name__ == "__main__":
    main()