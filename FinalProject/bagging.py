import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import csv
from helper import *

class Node:
    def __init__(self, children, attribute, label):
        self.children = children
        self.attribute = attribute
        self.label = label

class Purity:
    def __init__(self):
        pass

    def purity(self, label_counts, total_count):
        pass

class InformationGain(Purity):
    def purity(self, label_counts, total_count):
        probabilities = []
        for label, count in label_counts.items():
            probabilities.append(count / total_count)

        entropy = sum([p * math.log(p) for p in probabilities])
        return -entropy

def get_label_counts(examples):
    label_counts = {}

    counter = 0
    for example in examples:
        # count each label
        label = example["label"]
        if label in label_counts:
            label_counts[label] = label_counts[label] + example['weight']
        else:
            label_counts[label] = example['weight']

        counter += 1
    
    return label_counts

def sample_attributes(attributes, num_to_split_on):
    sample_attributes = []
    if num_to_split_on >= len(attributes):
        return attributes
    
    copy_of_attributes = []

    for attribute in attributes:
        copy_of_attributes.append(attribute)
    
    for i in range(num_to_split_on):
        index = random.randint(0, len(copy_of_attributes) - 1)
        sample_attributes.append(copy_of_attributes.pop(index))

    return sample_attributes
        

# ID3 algorithm to return a decision tree
def ID3(examples, attributes_and_vals, purity_measure, max_depth, num_to_split_on):
    attributes = list(attributes_and_vals.keys())
    dif_labels = False

    first_example = examples[0]
    first_label = first_example["label"]

    label_counts = get_label_counts(examples)

    # check if labels are all the same
    for example in examples:
        label = example["label"]
        if (label != first_label):
            dif_labels = True

    # Base case 1: labels are all the same
    if dif_labels == False:
        return Node(None, None, first_label)
    
    # Base case 2: no more attributes, or depth is exceeded
    if len(attributes) == 0 or max_depth == 0:
        best_count = 0
        best_label = ""

        for label, count in label_counts.items():
            if count > best_count:
                best_count = count
                best_label = label

        return Node(None, None, best_label)

    # create root node
    root_node = Node(None, None, None)

    total_length = 0

    for label, count in label_counts.items():
        total_length += count

    total_purity = purity_measure.purity(label_counts, total_length)

    subset_attributes = attributes

    if num_to_split_on != None:
        subset_attributes = sample_attributes(attributes, num_to_split_on)

    best_gain = 0
    best_attribute = subset_attributes[0]

    # find best attribute using the purity measure
    for attribute in subset_attributes:
        attribute_vals = {}

        counter = 0
        for example in examples:
            if example[attribute] in attribute_vals:
                attribute_vals[example[attribute]].append(example)
            else:
                attribute_vals[example[attribute]] = []
                attribute_vals[example[attribute]].append(example)
            
            counter += 1
        
        expectedPurity = 0
        for attribute_val, attr_examples in attribute_vals.items():
            new_label_counts = get_label_counts(attr_examples)

            partial_length = 0

            for label, count in new_label_counts.items():
                partial_length += count

            purity = purity_measure.purity(new_label_counts, partial_length)
            weighted_purity = partial_length / total_length * purity
            expectedPurity += weighted_purity
        
        gain = total_purity - expectedPurity

        if (gain > best_gain):
            best_gain = gain
            best_attribute = attribute
    
    root_node.attribute = best_attribute

    children = {}

    for attribute_val in attributes_and_vals[best_attribute]:
        children[attribute_val] = []
    
    # split examples on the best_attribute found earlier
    for example in examples:
        attribute_val = example[best_attribute]
        children[attribute_val].append(example)

    children_of_root = {}

    for attribute_val, new_examples in children.items():
        # set is empty, create a label using the majority label
        if (len(new_examples) == 0):
            best_count = 0
            best_label = ""

            for label, count in label_counts.items():
                if count > best_count:
                    best_count = count
                    best_label = label
            children_of_root[attribute_val] = Node(None, None, best_label)
        # recurse on the new set
        else:
            new_attributes_and_vals = {}
            for attr, vals in attributes_and_vals.items():
                if (attr != best_attribute):
                    new_attributes_and_vals[attr] = vals
            children_of_root[attribute_val] = ID3(new_examples, new_attributes_and_vals, purity_measure, max_depth - 1, num_to_split_on)

    root_node.children = children_of_root
    return root_node

# updates unknown values with the majority label for that attribute
def update_unknown_values(train_data, test_data, attributes):
    majority_values = {}

    for name, vals in attributes.items():
        majority_values[name] = {}
        for val in vals:
            majority_values[name][val] = 0

    for example in train_data:
        for name, val in example.items():
            if name != 'label':
                if val != '?':
                    majority_values[name][val] = majority_values[name][val] + 1

    for attr, val_counts in majority_values.items():
        best_count = 0
        best_val = ""
        for val, count in val_counts.items():
            if count > best_count:
                best_count = count
                best_val = val
        
        majority_values[attr] = best_val

    for example in train_data:
        for attr, val in example.items():
            if val == '?':
                example[attr] = majority_values[attr]

    for example in test_data:
        for attr, val in example.items():
            if val == '?':
                example[attr] = majority_values[attr]

# get the prediction from the decision tree for the example
def predict(tree, example):
    current_subtree = tree
    while current_subtree.label == None:
        attr_val = example[current_subtree.attribute]
        current_subtree = current_subtree.children[attr_val]

    return current_subtree.label

# converts numeric data to categorical by comparing numeric values to median and 
# determining if the value is bigger or smaller than the median
def numeric_to_categorical(train_data, test_data, attributes):
    for name, vals in attributes.items():
        if vals[0] == 'numeric':
            numeric_vals = []
            for example in train_data:
                numeric_vals.append(float(example[name]))
            
            numeric_vals = sorted(numeric_vals)
            mid = int(len(numeric_vals) / 2)
            median = numeric_vals[mid]
            if len(numeric_vals) % 2 == 0:
                median = (median + numeric_vals[mid - 1]) / 2

            for example in train_data:
                if float(example[name]) > median:
                    example[name] = 'bigger'
                else:
                    example[name] = 'smaller'

            for example in test_data:
                if float(example[name]) > median:
                    example[name] = 'bigger'
                else:
                    example[name] = 'smaller'
            
            attributes[name] = ['bigger', 'smaller']

def bag(train_data, attributes, num_trees, num_samples, num_to_split_on):
    trees = []
    for iteration in range(num_trees):
        new_examples = []
        for sample in range(num_samples):
            index = random.randint(0, len(train_data) - 1)
            new_examples.append(train_data[index])

        tree = ID3(new_examples, attributes, InformationGain(), -1, num_to_split_on)
        trees.append(tree)
    
    return trees

def train_via_decision_tree():
    attributes = read_attributes()
    attribute_names = list(attributes.keys())
    train_data = read_examples("income2023f/train_final.csv", attribute_names)
    test_data = read_examples("income2023f/test_final.csv", attribute_names)

    numeric_to_categorical(train_data, test_data, attributes)

    attributes.pop('label')

    update_unknown_values(train_data, test_data, attributes)

    for example in train_data:
        example['weight'] = 1

    for example in test_data:
        example['weight'] = 1
    
    tree = ID3(train_data, attributes, InformationGain(), -1, len(attributes))

    labels = []
    for example in test_data:
        labels.append(predict(tree, example))

    save_csv(labels, "decision_tree")

def bagged_prediction(trees, example):
    prediction_counts = {}
    for tree in trees:
        prediction = predict(tree, example)
        if prediction in prediction_counts:
            prediction_counts[prediction] += 1
        else:
            prediction_counts[prediction] = 1

    best_count = 0
    final_prediction = None        
    
    for prediction, count in prediction_counts.items():
        if count > best_count:
            best_count = count
            final_prediction = prediction

    return final_prediction

# determines percent of correctly predicted labels
def bagging_error(trees, examples):
    num_examples = len(examples)
    num_right = 0
    for example in examples:
        prediction = bagged_prediction(trees, example)
        if prediction == example["label"]:
            num_right = num_right + 1

    return 1 - (num_right / num_examples)

def train_via_bagging():
    attributes = read_attributes()
    attribute_names = list(attributes.keys())
    train_data = read_examples("income2023f/train_final.csv", attribute_names)
    test_data = read_examples("income2023f/test_final.csv", attribute_names)

    numeric_to_categorical(train_data, test_data, attributes)

    attributes.pop('label')

    update_unknown_values(train_data, test_data, attributes)

    for example in train_data:
        example['weight'] = 1

    for example in test_data:
        example['weight'] = 1
    
    trees = bag(train_data, attributes, num_trees=500, num_samples=1000, num_to_split_on=None)

    labels = []
    for example in test_data:
        labels.append(bagged_prediction(trees, example))

    save_csv(labels, "bagged_submission")

def main():
    #train_via_decision_tree()
    train_via_bagging()

if __name__ == "__main__":
    main()