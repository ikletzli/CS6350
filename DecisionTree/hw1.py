import os
import math

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

class Node:
    def __init__(self, children, attribute, label):
        self.children = children
        self.attribute = attribute
        self.label = label

    # Instance method
    def method1(self):
        print("")

def get_label_counts(examples):
    label_counts = {}

    for example in examples:
        # count each label
        label = example["label"]
        if label in label_counts:
            label_counts[label] = label_counts[label] + 1
        else:
            label_counts[label] = 1
    
    return label_counts

def ID3(examples, attributes_and_vals, purity_measure, max_depth):
    attributes = list(attributes_and_vals.keys())
    dif_labels = False

    first_example = examples[0]
    first_label = first_example["label"]

    label_counts = get_label_counts(examples)

    for example in examples:
        label = example["label"]
        if (label != first_label):
            dif_labels = True

    if dif_labels == False:
        return Node(None, None, first_label)
    
    if len(attributes) == 0 or max_depth == 0:
        best_count = 0
        best_label = ""

        for label, count in label_counts.items():
            if count > best_count:
                best_count = count
                best_label = label

        return Node(None, None, best_label)

    root_node = Node(None, None, None)

    total_purity = purity_measure.purity(label_counts, len(examples))

    best_gain = 0
    best_attribute = attributes[0]

    for attribute in attributes:
        attribute_vals = {}
        for example in examples:
            if example[attribute] in attribute_vals:
                attribute_vals[example[attribute]].append(example)
            else:
                attribute_vals[example[attribute]] = []
                attribute_vals[example[attribute]].append(example)
        
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
    
    for example in examples:
        attribute_val = example[best_attribute]
        children[attribute_val].append(example)

    children_of_root = {}

    for attribute_val, new_examples in children.items():
        if (len(new_examples) == 0):
            best_count = 0
            best_label = ""

            for label, count in label_counts.items():
                if count > best_count:
                    best_count = count
                    best_label = label
            children_of_root[attribute_val] = Node(None, None, best_label)
        else:
            new_attributes_and_vals = {}
            for attr, vals in attributes_and_vals.items():
                if (attr != best_attribute):
                    new_attributes_and_vals[attr] = vals
            children_of_root[attribute_val] = ID3(new_examples, new_attributes_and_vals, purity_measure, max_depth - 1)

    root_node.children = children_of_root
    return root_node

def read_examples(file_name, attributes):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, file_name)
    examples = []

    with open (file_path, 'r') as f:
        for line in f:
            values = line.strip().split(',')
            example = {}
            for i in range(len(attributes)):
                example[attributes[i]] = values[i]

            examples.append(example)
    
    return examples

def read_description(file_name):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, file_name)

    attributes = {}

    with open (file_path, 'r') as f:
        for line in f:
            if "attributes" in line:
                f.readline()

                for i in range(6):
                    attribute = f.readline()
                    values = attribute.strip().split(':')
                    values[1] = values[1].strip()
                    values[1] = values[1][:len(values[1]) - 1]
                    values[1] = values[1].split(', ')
                    attributes[values[0]] = values[1]
    
    return attributes

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

def train_tree(train_data, purity_measure, max_depth):
    attributes = read_description("car/data-desc.txt")
    tree = ID3(train_data, attributes, purity_measure, max_depth)
    return tree

def predict(tree, example):
    current_subtree = tree
    while current_subtree.label == None:
        attr_val = example[current_subtree.attribute]
        current_subtree = current_subtree.children[attr_val]

    return current_subtree.label

def percent_predicted_correct(tree, examples):
    num_examples = len(examples)
    num_right = 0
    for example in examples:
        prediction = predict(tree, example)
        if prediction == example["label"]:
            num_right = num_right + 1

    return num_right / num_examples

def evaluate_car_tree(purity_measures):
    attributes = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]
    train_data = read_examples("car/train.csv", attributes)
    test_data = read_examples("car/test.csv", attributes)

    attributes = read_description("car/data-desc.txt")
    for purity_measure in purity_measures:
        for depth in range(6):
            tree = ID3(train_data, attributes, purity_measure, depth + 1)
            test_err = percent_predicted_correct(tree, test_data)
            train_err = percent_predicted_correct(tree, train_data)

            class_name = purity_measure.__class__.__name__
            print(f"Purity: {class_name}, Depth: {depth + 1}, Test Error: {test_err}, Train Error: {train_err}")
        
        print("")

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

def evaluate_bank_tree(purity_measures):

    attributes = read_bank_description("bank/data-desc.txt")
    attribute_names = list(attributes.keys())
    train_data = read_examples("bank/train.csv", attribute_names)
    test_data = read_examples("bank/test.csv", attribute_names)

    numeric_to_categorical(train_data, test_data, attributes)

    attributes.pop('label')

    for purity_measure in purity_measures:
        for depth in range(16):
            tree = ID3(train_data, attributes, purity_measure, depth + 1)
            test_err = percent_predicted_correct(tree, test_data)
            train_err = percent_predicted_correct(tree, train_data)

            class_name = purity_measure.__class__.__name__
            print(f"Purity: {class_name}, Depth: {depth + 1}, Test Error: {test_err}, Train Error: {train_err}")
        
        print("")

    update_unknown_values(train_data, test_data, attributes)

    for purity_measure in purity_measures:
        for depth in range(16):
            tree = ID3(train_data, attributes, purity_measure, depth + 1)
            test_err = percent_predicted_correct(tree, test_data)
            train_err = percent_predicted_correct(tree, train_data)

            class_name = purity_measure.__class__.__name__
            print(f"Purity: {class_name}, Depth: {depth + 1}, Test Error: {test_err}, Train Error: {train_err}")
        
        print("")

def main():
    purity_measures = []
    purity_measures.append(GiniIndex())
    purity_measures.append(MajorityError())
    purity_measures.append(InformationGain())

    evaluate_bank_tree(purity_measures)
    evaluate_car_tree(purity_measures)

if __name__ == "__main__":
    main()