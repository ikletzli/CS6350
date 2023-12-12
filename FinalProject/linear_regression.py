import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import random

# converts the examples into a list of maps that map an example's attributes to its values
def read_examples(file_name, attributes):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, file_name)
    examples = []

    with open (file_path, 'r') as f:
        for line in f:
            values = line.strip().split(',')
            example = np.zeros(8)
            for i in range(len(attributes)):
                example[i] = values[i]

            examples.append(example)
    
    return examples

# reads the attributes for the bank data
def read_concrete_description(file_name):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, file_name)

    attributes = []

    with open (file_path, 'r') as f:
        for line in f:
            if "Cement" in line:
                attributes.append(line[:-1])
                for i in range(6):
                    attributes.append(f.readline()[:-1])
    
    attributes.append('label')
    
    return attributes

def batch_gradient_descent(x, y):
    r = 0.01
    w = np.zeros_like(x[:,0])

    costs = []
    iterations = 1
    while True:
        cost = 0.5 * np.sum((y - w.dot(x))**2)
        costs.append(cost)

        grad = np.zeros_like(w)
        for i in range(w.size):
            grad[i] = -np.sum((y - w.dot(x)) * x[i,:])

        new_w = w - r*grad
        if np.linalg.norm(new_w - w) < 1e-6:
            w = new_w
            break

        w = new_w
        iterations += 1

    return costs, w, r

def stochastic_gradient_descent(x, y):
    r = 5e-3
    w = np.zeros_like(x[:,0])

    costs = []
    iterations = 1
    while True:
        index = random.randint(0, y.size - 1)
        new_x = x[:,index]
        new_y = y[index]
        cost = 0.5 * np.sum((y - w.dot(x))**2)
        costs.append(cost)

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
    attributes = read_concrete_description("concrete/data-desc.txt")
    train_data = read_examples("concrete/train.csv", attributes)
    test_data = read_examples("concrete/test.csv", attributes)

    x = np.zeros((8, len(train_data)))
    x[0,:] = np.ones(len(train_data))
    y = np.zeros(len(train_data))

    counter = 0
    for example in train_data:
        x[1:, counter] = example[0:7]
        y[counter] = example[7]
        counter += 1

    b_costs, b_w, b_r = batch_gradient_descent(x, y)
    s_costs, s_w, s_r = stochastic_gradient_descent(x, y)

    analytical_optimal_w = np.linalg.inv(x.dot(x.T)).dot(x).dot(y) 

    print(f"Analytical optimal w: {analytical_optimal_w}\n")
    print(f"W from batch gradient descent using a learning rate of {b_r}: {b_w}\n")
    print(f"W from stochastic gradient descent using a learning rate of {s_r}: {s_w}\n")

    x = np.zeros((8, len(test_data)))
    x[0,:] = np.ones(len(test_data))
    y = np.zeros(len(test_data))

    counter = 0
    for example in test_data:
        x[1:, counter] = example[0:7]
        y[counter] = example[7]
        counter += 1

    b_test_cost = 0.5 * np.sum((y - b_w.dot(x))**2)
    s_test_cost = 0.5 * np.sum((y - s_w.dot(x))**2)

    print(f"Cost function value of test data for stochastic gradient descent: {s_test_cost}")
    print(f"Cost function value of test data for batch gradient descent: {b_test_cost}")


    plt.title("Cost for stochastic gradient descent")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")

    plt.plot(np.array(range(len(s_costs))), np.array(s_costs), color='b')
    plt.show()

    plt.title("Cost for batch gradient descent")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")

    plt.plot(np.array(range(len(b_costs))), np.array(b_costs), color='b')
    plt.show()

def main():
    lms()

if __name__ == "__main__":
    main()