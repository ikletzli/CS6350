import os
import math
import matplotlib.pyplot as plt
import numpy as np
import random

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

# converts the examples into a list of maps that map an example's attributes to its values
def read_examples(file_name, attributes):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, file_name)
    examples = []

    with open (file_path, 'r') as f:
        for line in f:
            values = line.strip().split(',')
            example = {}
            for i in range(len(attributes)):
                if attributes[i] == 'label':
                    if values[i] == 'no':
                        example[attributes[i]] = -1
                    else:
                        example[attributes[i]] = 1

                else:
                    example[attributes[i]] = values[i]

            examples.append(example)
    
    return examples

# reads the attributes for the bank data
def read_bank_description(file_name):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, file_name)

    attributes = {}

    with open (file_path, 'r') as f:
        for line in f:
            if "-" in line:
                index = line.index("-")
                line = line[index+2:]

                values = line.strip().split(':')
                values[0] = values[0].strip()

                if "numeric" in line:
                    attr_vals = values[1]
                    attributes[values[0]] = ['numeric']
                else:
                    attr_vals = values[2]
                    index_of_quote = attr_vals.index('"')
                    attr_vals = attr_vals[index_of_quote:len(attr_vals) - 1]
                    attr_vals = attr_vals.replace('"', '')
                    attr_vals = attr_vals.split(',')
                    attributes[values[0]] = attr_vals
    
    return attributes

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

def new_problem():
    attributes = read_bank_description("bank/data-desc.txt")
    attribute_names = list(attributes.keys())
    train_data = read_examples("bank/train.csv", attribute_names)
    test_data = read_examples("bank/test.csv", attribute_names)

    for example in train_data:
        example['weight'] = 1

    for example in test_data:
        example['weight'] = 1

    numeric_to_categorical(train_data, test_data, attributes)

    attributes.pop('label')

    print("Evalutating bank data with 'unknown' as an attribute value:\n")

    bagged_predictors = []
    single_trees = []

    for i in range(100):
        trees = bag(train_data, attributes, num_trees=500, num_samples=1000, num_to_split_on=None)
        bagged_predictors.append(trees)
        print(f"Iteration {i + 1} of bagging")

    for predictor in bagged_predictors:
        single_trees.append(predictor[0])

    single_tree_biases = []
    single_tree_variances = []
    bagged_biases = []
    bagged_variances = []

    for example in test_data:
        # compute bias and variance for single trees
        predictions = []
        for tree in single_trees:
            predictions.append(predict(tree, example))

        predictions = np.array(predictions)
        average = np.sum(predictions) / predictions.size
        bias = (average - example['label']) ** 2
        variance = (1 / (predictions.size - 1)) * np.sum((predictions-average)**2)
        single_tree_biases.append(bias)
        single_tree_variances.append(variance)

        # compute bias and variance for bagged predictors
        predictions = []
        for bagged_predictor in bagged_predictors:
            predictions.append(bagged_prediction(bagged_predictor, example))

        predictions = np.array(predictions)
        average = np.sum(predictions) / predictions.size
        bias = (average - example['label']) ** 2
        variance = (1 / (predictions.size - 1)) * np.sum((predictions-average)**2)
        bagged_biases.append(bias)
        bagged_variances.append(variance)

    single_tree_biases = np.array(single_tree_biases)
    single_tree_variances = np.array(single_tree_variances)
    bagged_biases = np.array(bagged_biases)
    bagged_variances = np.array(bagged_variances)

    average_single_tree_bias = np.sum(single_tree_biases) / single_tree_biases.size
    average_single_tree_variance = np.sum(single_tree_variances) / single_tree_variances.size
    error_estimate_st = average_single_tree_bias + average_single_tree_variance

    average_bagged_bias = np.sum(bagged_biases) / bagged_biases.size
    average_bagged_variance = np.sum(bagged_variances) / bagged_variances.size
    error_estimate_bag = average_bagged_bias + average_bagged_variance

    print(f"Single Tree Estimates:\n Bias: {average_single_tree_bias:.4f}, Variance: {average_single_tree_variance:.4f}, Error: {error_estimate_st:.4f}")
    print(f"Bagged Estimates:\n Bias: {average_bagged_bias:.4f}, Variance: {average_bagged_variance:.4f}, Error: {error_estimate_bag:.4f}")


def evaluate_bank_tree():
    attributes = read_bank_description("bank/data-desc.txt")
    attribute_names = list(attributes.keys())
    train_data = read_examples("bank/train.csv", attribute_names)
    test_data = read_examples("bank/test.csv", attribute_names)

    for example in train_data:
        example['weight'] = 1

    for example in test_data:
        example['weight'] = 1

    numeric_to_categorical(train_data, test_data, attributes)

    attributes.pop('label')

    print("Evalutating bank data with 'unknown' as an attribute value:\n")

    xpoints = []
    test_errs = []
    train_errs = []

    for num_trees in range(500):
        xpoints.append(num_trees + 1)
        trees = bag(train_data, attributes, num_trees=num_trees+1, num_samples=1000, num_to_split_on=None)
        train_err = bagging_error(trees, train_data)
        train_errs.append(train_err)
        test_err = bagging_error(trees, test_data)
        test_errs.append(test_err)
        print(f"Number of Trees: {num_trees}, Test Error: {test_err:.4f}, Train Error: {train_err:.4f}")

    xpoints = np.array(xpoints)

    plt.title("Test and Train Error for Bagging")
    plt.xlabel("Number of Trees")
    plt.ylabel("Error")

    plt.plot(xpoints, np.array(test_errs), color='r', label='test')
    plt.plot(xpoints, np.array(train_errs), color='b', label='train')
    plt.legend()
    plt.show()


def random_forest():
    attributes = read_bank_description("bank/data-desc.txt")
    attribute_names = list(attributes.keys())
    train_data = read_examples("bank/train.csv", attribute_names)
    test_data = read_examples("bank/test.csv", attribute_names)

    for example in train_data:
        example['weight'] = 1

    for example in test_data:
        example['weight'] = 1

    numeric_to_categorical(train_data, test_data, attributes)

    attributes.pop('label')

    print("Evalutating bank data with 'unknown' as an attribute value:\n")

    xpoints = []
    test_errs = {2:[],4:[],6:[]}
    train_errs = {2:[],4:[],6:[]}

    for num_trees in range(20):
        xpoints.append(num_trees + 1)
        for split in [2,4,6]:
            trees = bag(train_data, attributes, num_trees=num_trees+1, num_samples=1000, num_to_split_on=split)
            train_err = bagging_error(trees, train_data)
            train_errs[split].append(train_err)
            test_err = bagging_error(trees, test_data)
            test_errs[split].append(test_err)
            print(f"Number of Trees: {num_trees}, Test Error: {test_err:.4f}, Train Error: {train_err:.4f}, Random Forest Split: {split}")

    xpoints = np.array(xpoints)

    plt.title("Test and Train Error for Random Forests")
    plt.xlabel("Number of Trees")
    plt.ylabel("Error")

    plt.plot(xpoints, np.array(test_errs[2]), label='test, split=2')
    plt.plot(xpoints, np.array(train_errs[2]), label='train, split=2')

    plt.plot(xpoints, np.array(test_errs[4]), label='test, split=4')
    plt.plot(xpoints, np.array(train_errs[4]), label='train, split=4')

    plt.plot(xpoints, np.array(test_errs[6]), label='test, split=6')
    plt.plot(xpoints, np.array(train_errs[6]), label='train, split=6')
    plt.legend()
    plt.show()

def main():
    #evaluate_bank_tree()
    #new_problem()
    random_forest()

if __name__ == "__main__":
    main()