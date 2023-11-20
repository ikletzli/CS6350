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

def schedule(l0, t, a):
    return l0 / (1 + (l0/a) * t)

def svm_gradient_descent(x, num_epochs, C, l0, a):
    w = np.zeros((5,))
    
    gamma_t = l0

    for T in range(num_epochs):
        np.random.shuffle(x)
        for example in x:
            y = example[5]
            this_x = example[0:5]
            grad_w = np.zeros((5,))
            grad_w[1:5] = w[1:5]
            if (y * w.dot(this_x) <= 1):
                w = w - gamma_t * grad_w + gamma_t * C * x.shape[0] * y * this_x
            else:
                w = w - gamma_t * grad_w

        #print(1-x[:,5]*x[:,0:5].dot(w))
        #print(np.sum(np.fmax(0, 1-x[:,5]*x[:,0:5].dot(w))))
        #print(x[:,0:5].shape)
        #print(x[:,5].shape)
        #print("Iteration:", T+1, 0.5 * w.dot(w) + C * np.sum(np.fmax(0, 1-x[:,5]*x[:,0:5].dot(w))))
        gamma_t = schedule(l0, T+1, a)

    return w

def evaluate_perceptron():
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

    l0 = [5e-5, 3e-6]
    a = [5e-5, 6e-3]

    x_test = x_test.T
    for i in range(2):
        for C in [100/873, 500/873, 700/873]:
            w = svm_gradient_descent(x_train,num_epochs=100, C=C, l0=l0[i], a=a[i])
            test_err = perceptron_prediction(x_test[:,0:5], x_test[:,5], w)
            train_err = perceptron_prediction(x_train[:,0:5], x_train[:,5], w)

            print("Test Error:", test_err, "Train Error:", train_err)


    # w = vanilla_perceptron(x_train,r=0.1,num_epochs=10)
    # vanilla_err = perceptron_prediction(x_test, y_test, w)
    # print(f"Weight vector for vanilla perceptron: \n\t{w}")
    # print(f"Prediction error on test set for vanilla perceptron: \n\t{vanilla_err}\n")

    # return_vals = voted_perceptron(x_train,y_train,r=0.1,num_epochs=10)
    # voted_err = voted_perceptron_prediction(x_test, y_test, return_vals)
    # print(f"Weight vectors and counts for voted perceptron:")
    # for weight, count in return_vals:
    #     print(f"\tWeight: {weight} Count: {count}")

    # print(f"Prediction error on test set for voted perceptron: \n\t{voted_err}\n")

    # a = average_perceptron(x_train,y_train,r=0.1,num_epochs=10)
    # average_err = perceptron_prediction(x_test, y_test, a)
    # print(f"Weight vector for average perceptron: \n\t{a}")
    # print(f"Prediction error on test set for average perceptron: \n\t{average_err}")

def main():
    evaluate_perceptron()

if __name__ == "__main__":
    main()