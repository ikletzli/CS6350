import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import csv

# converts the examples into a list of maps that map an example's attributes to its values
def read_examples(file_name, attributes):
    is_test = "test" in file_name
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, file_name)
    examples = []

    with open (file_path, 'r') as f:
        f.readline()
        for line in f:
            values = line.strip().split(',')
            example = {}
            for i in range(len(attributes)):
                if attributes[i] == 'label':
                    if is_test or values[i] == '0':
                        example[attributes[i]] = -1
                    else:
                        example[attributes[i]] = 1

                else:
                    if is_test:
                        example[attributes[i]] = values[i+1]
                    else:
                        example[attributes[i]] = values[i]

            examples.append(example)
    
    return examples

# reads the attributes for the income data
def read_attributes():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, "income2023f/description.txt")

    attributes = {}

    with open (file_path, 'r') as f:
        for line in f:
            values = line.strip().split(': ')
            values[0] = values[0].strip()

            if "continuous" in line:
                attr_vals = values[1]
                attributes[values[0]] = ['numeric']
            else:
                attr_vals = values[1][1:-1]
                attr_vals = attr_vals.split(", ")
                attributes[values[0]] = attr_vals
    
    return attributes

def save_csv(labels, file_name):
    with open(f'{file_name}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Prediction"])
        for i in range(len(labels)):
            writer.writerow([i+1, labels[i]])