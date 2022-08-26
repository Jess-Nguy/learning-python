"""
This program is designed to use perceptron learning alogrithm to classify numeric data on 4 different files to see if it's lineraly separable.

Q: Using the results you got, which
sets would you say are linearly separable and which are not linearly separable?
A: files 1 and 2 are linearly separable. 3 and 4 are linearly separable because the first 2 got 100% accuracy on testing and the others were less than 50% accuracy with test data.

Jess Nguyen 
"""

import numpy as np
import csv


def perceptron(weights, threshold, target, data):
    """
    Used to get the perceptron by number of epochs.
    """
    for epoch in range(0, 100):
        outputArr = []
        for input in range(0, len(data)):
            activation = np.array(
                (weights) * np.array(data[input])).sum() > np.array(threshold)
            if activation:
                outputArr.append(1)
                output = 1
            else:
                outputArr.append(0)
                output = 0
            if output < target[input]:
                weights = (np.array(weights) +
                           np.array(data[input]) * learningRate)
                threshold = threshold - learningRate
            elif output > target[input]:
                weights = (np.array(weights) -
                           np.array(data[input]) * learningRate)
                threshold = threshold + learningRate
        rights = (outputArr == target)
        trainAccuracy = round(np.array(rights).sum()/len(outputArr)*100, 2)
    return weights, threshold


def perceptronAccuracyBreak(weights, threshold, target, data):
    """
    Used to get perceptron by accuracy cap and falls back on number of epochs if unattainable.
    """
    underAcc = 0
    while underAcc <= 300:
        outputArr = []
        for input in range(0, len(data)):
            activation = np.array(
                (weights) * np.array(data[input])).sum() > np.array(threshold)
            if activation:
                outputArr.append(1)
                output = 1
            else:
                outputArr.append(0)
                output = 0
            if output < target[input]:
                weights = (np.array(weights) +
                           np.array(data[input]) * learningRate)
                threshold = threshold - learningRate
            elif output > target[input]:
                weights = (np.array(weights) -
                           np.array(data[input]) * learningRate)
                threshold = threshold + learningRate
        rights = (outputArr == target)
        trainAccuracy = round(np.array(rights).sum()/len(outputArr)*100, 2)
        if(trainAccuracy >= 100):
            underAcc = 301
        underAcc += 1
    return weights, threshold


def prediction(weights, threshold, data, target):
    """
    Used to get predictions based on training data parameters(weight and threshold) on testing data and testing target.
    """
    results = []
    for input in range(0, len(data)):
        activation = np.array(
            (weights) * np.array(data[input])).sum() > np.array(threshold)
        if activation:
            results.append(1)
        else:
            results.append(0)
    correct = (results == target)
    accuracy = round(np.array(correct).sum()/len(results)*100)
    return accuracy


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

    trainingWeights = np.random.uniform(-1, 1, np.array(trainingData).shape[1])

    return trainingData, trainingTarget, testingData, testingTarget, trainingWeights


# Uses perceptron to train and then test the data for each file.
files = ['jess_nguyen_1.csv', 'jess_nguyen_2.csv',
         'jess_nguyen_3.csv', 'jess_nguyen_4.csv']
for f in files:

    trainingData, trainingTarget, testingData, testingTarget, trainingWeights = fileReadSplit(
        f)

    learningRate = 0.1
    trainingThreshold = 0

# Command for epoch perceptron function call
    # trainingWeights, trainingThresold = perceptron(trainingWeights, trainingThreshold, trainingTarget, trainingData)

    trainingWeights, thresold = perceptronAccuracyBreak(
        trainingWeights, trainingThreshold, trainingTarget, trainingData)

    accuracy = prediction(trainingWeights, trainingThreshold,
                          testingData, testingTarget)

    print("\n" + f + ":", str(accuracy) + "%", "W:",
          np.round(trainingWeights, 1), "T:", round(thresold, 1))
