# CS6350

This is a machine learning library developed by Isaac Kletzli for
CS5350/6350 in University of Utah

For HW2 (Bagging, Boosting, and Linear Regression) there are 3 files to be aware of:

1. EnsembleLearning/bagging.py
2. EnsembleLearning/boosting.py
3. LinearRegression/linear_regression.py

To run the bagging file, start in the top level directory and run: python3 ./EnsembleLearning/bagging.py [num_trees] [num_iterations] where num_trees is the number of trees to run the bagging and random forest algorithms on, and where num_iterations is the number of times to collect data for calculating the bias and variance. The default value for num_trees is 20 and the default value for num_iterations is 2.

To run the boosting file, start in the top level directory and run: python3 ./EnsembleLearning/boosting.py [num_iterations] where num_iterations is the number of iterations to run AdaBoost for. The default value for num_iterations is 20.

To run the linear regression file, start in the top level directory and run: python3 ./LinearRegression/linear_regression.py.

To run perceptron for HW3, start in the top level directory and run: python3 ./Perceptron/perceptron.py.

To run svm for HW4, start in the top level directory and run: python3 ./SVM/svm.py.

To run the neural networks for HW5, start in the top level directory and run: python3 ./NeuralNetworks/neural_networks.py.
