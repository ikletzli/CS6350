import os
import sys
import math
#import matplotlib.pyplot as plt
import numpy as np
import random

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
    train_data = read_examples("bank-note/train.csv")
    test_data = read_examples("bank-note/test.csv")

    x_train = np.zeros((5, len(train_data)))
    x_train[0,:] = np.ones(len(train_data))
    y_train = np.zeros(len(train_data))

    counter = 0
    for example in train_data:
        x_train[1:, counter] = example[0:4]
        y_train[counter] = example[4]
        counter += 1

    x_train = x_train.T

    x_test = np.zeros((5, len(test_data)))
    x_test[0,:] = np.ones(len(test_data))
    y_test = np.zeros(len(test_data))

    counter = 0
    for example in test_data:
        x_test[1:, counter] = example[0:4]
        y_test[counter] = example[4]
        counter += 1

    x_test = x_test.T

    w = vanilla_perceptron(x_train,y_train,r=0.1,num_epochs=10)
    vanilla_err = perceptron_prediction(x_test, y_test, w)
    print(f"Weight vector for vanilla perceptron: \n\t{w}")
    print(f"Prediction error on test set for vanilla perceptron: \n\t{vanilla_err}\n")

    return_vals = voted_perceptron(x_train,y_train,r=0.1,num_epochs=10)
    voted_err = voted_perceptron_prediction(x_test, y_test, return_vals)
    print(f"Weight vectors and counts for voted perceptron:")
    for weight, count in return_vals:
        print(f"\tWeight: {weight} Count: {count}")

    print(f"Prediction error on test set for voted perceptron: \n\t{voted_err}\n")

    a = average_perceptron(x_train,y_train,r=0.1,num_epochs=10)
    average_err = perceptron_prediction(x_test, y_test, a)
    print(f"Weight vector for average perceptron: \n\t{a}")
    print(f"Prediction error on test set for average perceptron: \n\t{average_err}")

def main():
    evaluate_perceptron()

if __name__ == "__main__":
    main()