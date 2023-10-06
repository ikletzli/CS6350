import os
import math
import matplotlib.pyplot as plt 

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

class MajorityError(Purity):
    def purity(self, label_counts, total_count):
        best_count = 0

        for label, count in label_counts.items():
            if count > best_count:
                best_count = count

        return (total_count - best_count) / total_count
    
class GiniIndex(Purity):
    def purity(self, label_counts, total_count):
        probabilities = []
        for label, count in label_counts.items():
            probabilities.append(count / total_count)

        sum_prob_square = sum([p * p for p in probabilities])
        return 1.0 - sum_prob_square

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

# ID3 algorithm to return a decision tree
def ID3(examples, attributes_and_vals, purity_measure, max_depth):
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

    total_purity = purity_measure.purity(label_counts, len(examples))

    best_gain = 0
    best_attribute = attributes[0]

    # find best attribute using the purity measure
    for attribute in attributes:
        attribute_vals = {}
        new_weights = []

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
            purity = purity_measure.purity(new_label_counts, len(attr_examples))
            weighted_purity = len(attr_examples) / len(examples) * purity
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
            children_of_root[attribute_val] = ID3(new_examples, new_attributes_and_vals, purity_measure, max_depth - 1)

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

# determines percent of correctly predicted labels
def percent_predicted_correct(tree, examples):
    num_examples = len(examples)
    num_right = 0
    for example in examples:
        prediction = predict(tree, example)
        if prediction == example["label"]:
            num_right = num_right + 1

    return 1 - (num_right / num_examples)

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

# updates unknown values with the majority label for that attribute
def update_unknown_values(train_data, test_data, attributes):
    majority_values = {}

    for name, vals in attributes.items():
        if 'unknown' in vals:
            vals.remove('unknown')

    for name, vals in attributes.items():
        majority_values[name] = {}
        for val in vals:
            majority_values[name][val] = 0

    for example in train_data:
        for name, val in example.items():
            if name != 'label':
                if val != 'unknown':
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
            if val == 'unknown':
                example[attr] = majority_values[attr]

    for example in test_data:
        for attr, val in example.items():
            if val == 'unknown':
                example[attr] = majority_values[attr]

def print_err(purity_measures, train_data, attributes, test_data, max_depth):
    for purity_measure in purity_measures:
        class_name = purity_measure.__class__.__name__
        print(f"Purity: {class_name}")

        average_test_err = 0
        average_train_err = 0
        for depth in range(max_depth):
            tree = ID3(train_data, attributes, purity_measure, depth + 1)
            test_err = percent_predicted_correct(tree, test_data)
            train_err = percent_predicted_correct(tree, train_data)

            average_test_err += test_err
            average_train_err += train_err

            class_name = purity_measure.__class__.__name__
            print(f"Depth: {depth + 1}, Test Error: {test_err:.4f}, Train Error: {train_err:.4f}")

        print("")
        print(f"Average Test Error: {(average_test_err/max_depth):.4f}, Average Train Error: {(average_train_err/max_depth):.4f}")
        print("")

def ada_prediction(votes, classifiers, example):
    sum = 0
    for i in range(len(classifiers)):
        sum += votes[i] * predict(classifiers[i], example)

    sign = example['label'] * sum

    if sign > 0:
        return 1
    
    return 0

def print_err_for_one_depth(train_data, test_data, votes, classifiers, t):
    num_correct = 0
    for example in train_data:
        num_correct += ada_prediction(votes, classifiers, example)

    train_err = 1 - (num_correct / len(train_data))

    num_correct = 0
    for example in test_data:
        num_correct += ada_prediction(votes, classifiers, example)

    test_err = 1 - (num_correct / len(test_data))

    print(f"T: {t}, Test Error: {test_err:.4f}, Train Error: {train_err:.4f}")

def evaluate_bank_tree():
    attributes = read_bank_description("bank/data-desc.txt")
    attribute_names = list(attributes.keys())
    train_data = read_examples("bank/train.csv", attribute_names)
    test_data = read_examples("bank/test.csv", attribute_names)

    numeric_to_categorical(train_data, test_data, attributes)

    attributes.pop('label')

    print("Evalutating bank data with 'unknown' as an attribute value:\n")

    for t in range(500):
        votes, classifiers = ada_boost(train_data, attributes, t + 1)
        print_err_for_one_depth(train_data, test_data, votes, classifiers, t)

def ada_boost(train_data, attributes, T):
    weights = [1 / len(train_data)] * len(train_data) 

    for i in range(len(train_data)):
        train_data[i]['weight'] = weights[i]
    
    classifiers = [None] * T 
    votes = [None] * T 
    for t in range(T):
        tree = ID3(train_data, attributes, InformationGain(), 1)
        classifiers[t] = tree

        err_sum = 0
        counter = 0
        for example in train_data:
            # updates for e_t
            yi_hxi = example['label'] * predict(tree, example)
            err_sum += weights[counter] * yi_hxi
            counter += 1
            
        err_t = 0.5 - 0.5 * err_sum
        votes[t] = 0.5 * math.log((1 - err_t) / err_t)

        weight_sum = 0
        counter = 0
        for example in train_data:
            # updates for D_t+1
            yi_hxi = example['label'] * predict(tree, example)
            weights[counter] *= math.exp(-votes[t] * yi_hxi)
            weight_sum += weights[counter]
            counter += 1

        weights = [x / weight_sum for x in weights]

        for i in range(len(train_data)):
            train_data[i]['weight'] = weights[i]

    return votes, classifiers

def main():
    evaluate_bank_tree()

if __name__ == "__main__":
    main()