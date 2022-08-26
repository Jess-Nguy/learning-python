"""
https://archive.ics.uci.edu/ml/datasets/Swarm+Behaviour

This program is uses Flocking csv from the swarm behaviour dataset, student number generated data CSVs. It uses multi-layer perceptron to classify the dataset.

Jess Nguyen
"""

import numpy as np
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


def fileReadSplit(fileName):
    """
    Reads data files and split the data and target into training and testing arrays.
    """
    data = []
    target = []
    with open(fileName) as file:
        reader = csv.reader(file, delimiter=",")
        # loop through lines and append to arrays by column names
        for row in reader:
            target.append(row[len(row)-1:len(row)])
            data.append(row[:-1])
    target = np.array(target).flatten()

    data = np.array(data).astype(float)
    target = target.astype(float)

    # Calculate how many rows is 75% and how many rows is 25%
    totalRows = len(data)
    trainingNumRows = round(totalRows * 0.75)

    # Get the first 75% of the data to be training
    trainingData = np.array(data[:trainingNumRows])
    trainingTarget = np.array(target[:trainingNumRows])

    # Get the last 25% of the data to be testing
    testingData = np.array(data[trainingNumRows:])
    testingTarget = np.array(target[trainingNumRows:])

    return trainingData, trainingTarget, testingData, testingTarget


def getMLP(mlp, trainingData, trainingTarget, testingData, testingTarget):
    """
    Fit the training data to multi-layer perceptron then predict testing data and return accuracy.
    """
    mlp.fit(trainingData, trainingTarget)

    # Test the Classifier
    y_pred = mlp.predict(testingData)
    accuracyMLP = accuracy_score(testingTarget, y_pred)
    return accuracyMLP


files = ['jess_nguyen_1.csv', 'jess_nguyen_2.csv',
         'jess_nguyen_3.csv', 'jess_nguyen_4.csv', 'Flocking.csv']
for f in files:
    trainingData, trainingTarget, testingData, testingTarget = fileReadSplit(f)

    clf = tree.DecisionTreeClassifier(criterion="gini")
    clf = clf.fit(trainingData, trainingTarget)

    # Test the Classifier
    predictionTree = clf.predict(testingData)
    accuracyTree = accuracy_score(testingTarget, predictionTree)

    # Display
    print("\nFile:", f)
    print("Decision Tree:", round((accuracyTree * 100), 2), "% Accuracy")

    lr = 0.0005
    maxIter = 1000
    # Create and train a Multi-Layer Perceptron
    if f == 'jess_nguyen_1.csv':

        mlp = MLPClassifier(learning_rate_init=lr, max_iter=maxIter)
        accuracyMLP = getMLP(
            mlp, trainingData, trainingTarget, testingData, testingTarget)
        # Display
        print("MLP: hidden layers = default", "LR =", lr)
        print(round((accuracyMLP * 100), 2),
              "% Accuracy,", "iterations=", mlp.n_iter_)

    elif f == 'jess_nguyen_2.csv':

        mlp = MLPClassifier(learning_rate_init=lr, max_iter=maxIter)
        accuracyMLP = getMLP(
            mlp, trainingData, trainingTarget, testingData, testingTarget)
        # Display
        print("MLP: hidden layers = default max_iter=1000", "LR =", lr)
        print(round((accuracyMLP * 100), 2),
              "% Accuracy,", "iterations=", mlp.n_iter_)

    elif f == 'jess_nguyen_3.csv':

        mlp = MLPClassifier(hidden_layer_sizes=[
                            50, 25, 10, 5, 2], learning_rate_init=lr, max_iter=maxIter, batch_size=1)
        accuracyMLP = getMLP(
            mlp, trainingData, trainingTarget, testingData, testingTarget)
        # Display
        print("MLP: hidden layers = [50,25,10,5,2]",
              "LR =", lr, "batch_size=1 max_iter=1000")
        print(round((accuracyMLP * 100), 2),
              "% Accuracy,", "iterations=", mlp.n_iter_)

    elif f == 'jess_nguyen_4.csv':
        mlp = MLPClassifier(hidden_layer_sizes=[
                            45, 22, 11, 5, 2], learning_rate_init=lr, max_iter=maxIter, batch_size=1)
        accuracyMLP = getMLP(
            mlp, trainingData, trainingTarget, testingData, testingTarget)
        # Display
        print("MLP: hidden layers = [45,22,11,5,2]",
              "LR =", lr, "batch_size=1 max_iter=1000")
        print(round((accuracyMLP * 100), 2),
              "% Accuracy,", "iterations=", mlp.n_iter_)
    else:

        mlp = MLPClassifier(hidden_layer_sizes=[
                            60, 30, 10], learning_rate_init=lr, max_iter=maxIter, activation='tanh', batch_size=5)
        accuracyMLP = getMLP(
            mlp, trainingData, trainingTarget, testingData, testingTarget)
        # Display
        print("MLP: hidden layers = [60,30,10]",
              "LR =", lr, "activation=tanh batch_size=5")
        print(round((accuracyMLP * 100), 2),
              "% Accuracy,", "iterations=", mlp.n_iter_)

    # delete arrays for next files
    del trainingData
    del trainingTarget
    del testingData
    del testingTarget
