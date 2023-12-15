import os
import csv

# converts the examples into a list of maps that map an example's attributes to its values
old_range = 0.5089215868662893 - 0.0001566625369523826
new_range = 1
script_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_directory, "logistic.csv")

with open(f'new_range.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "Prediction"])

    with open (file_path, 'r') as f:
        f.readline()
        id = 1
        for line in f:
            new_val = (((float(line.split(",")[1]) - 0.0001566625369523826) * new_range) / old_range)
            writer.writerow([id, new_val])
            id+=1