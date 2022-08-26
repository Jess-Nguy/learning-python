"""
Jess Nguyen

Reading csv of phising data set to run training and testing predictions against the dataset with kNN algorithm.
https://archive.ics.uci.edu/ml/datasets/Website+Phishing

October 13 2021
"""

import csv
import numpy as np
from matplotlib import pyplot as plt

phishingData = []
labels = []

## Read csv file
with open("./../PhishingData.csv") as file:
    reader = csv.DictReader(file, delimiter=",")
    # loop through lines and append to arrays by column names
    for row in reader:
        labels.append(int(row["Result"]))
        dataRow = [int(row["SFH"]),int(row["popUpWindow"]),int(row["SSLfinal_State"]),int(row["Request_URL"]),int(row["URL_of_Anchor"]),int(row["web_traffic"]),int(row["URL_Length"]),int(row["age_of_domain"]),int(row["having_IP_Address"])]
        phishingData.append(dataRow)
    featureNames = list(row.keys())

# Make lists into numpy array
featureNames = np.array(featureNames[1:10])
labels = np.array(labels)
phishingData = np.array(phishingData)

# Calculate how many rows is 75% and how many rows is 25%
totalRows = len(phishingData)
trainingNumRows = round(totalRows * 0.75)

# names of the classes
names = sorted(set(labels))


def classify(newItem, data, labels, k, variation):
    """
    Classifying algorithm with different distance algorithms for kNN.
    newItem = 1d array, 1 row of the testing data.
    data = 2d array, the whole training data set.
    labels = 1d array, classification of the training data set.
    k = integer, kNN value for the algorithm.
    variation = String, the type of kNN algorithm.
    """
    closestKnn = {}
    if variation == "Manhattan with z-score":
        distance = np.abs(newItem - data).sum(axis=1)
    else:
        distance = np.sqrt(np.sum((data - newItem)**2,axis=1))
    labelArray =  np.array(labels[distance.argsort()][:k])
    for label in labelArray:
        closestKnn[label] = closestKnn.get(label, 0) + 1
    highestLabel = max(closestKnn, key=closestKnn.get)
    return highestLabel

def dataZ(data):
    """
    Z-score algorithm for different datasets.
    data = 2d array, either the testing data set or the training dataset.
    """
    return (data.mean(axis= 0) - data) / data.std(axis=0)

variationTypes = ["Euclidean Distance", "Euclidean with z-score", "Manhattan with z-score"]

averageCollection = [];
for variation in variationTypes:
    averageArrK12 = [];
    averageArrK8 = [];
    averageArrK4 = [];

    for runs in range(0,5):
        # Shuffling the phishingData & labels
        arangeShuffle = np.arange( phishingData.shape[0] )
        np.random.shuffle( arangeShuffle )
        phishingData = phishingData[arangeShuffle]
        labels = labels[arangeShuffle]

        # Get the first 75% of the phishingData to be training
        trainingData = np.array(phishingData[:trainingNumRows])
        trainingLabels = np.array(labels[:trainingNumRows])

        # Get the last 25% of the phishingData to be testing
        testingData = np.array(phishingData[trainingNumRows:])
        testingLabels = np.array(labels[trainingNumRows:])

        predictionK12 = [];
        predictionK8 = [];
        predictionK4 = [];

        if variation != "Euclidean Distance":
            testingData = dataZ(testingData)
            trainingData = dataZ(trainingData)

        for row in testingData:
            predictionK12.append(classify(row, trainingData, trainingLabels, 12, variation))
            predictionK8.append(classify(row, trainingData, trainingLabels, 8, variation))
            predictionK4.append(classify(row, trainingData, trainingLabels, 4, variation))

        comparisonK12 = testingLabels == predictionK12
        comparisonK8 = testingLabels == predictionK8
        comparisonK4 = testingLabels == predictionK4

        averageArrK12.append(round(comparisonK12.sum()/len(testingLabels) * 100, 2));
        averageArrK8.append(round(comparisonK8.sum()/len(testingLabels) * 100, 2));
        averageArrK4.append(round(comparisonK4.sum()/len(testingLabels) * 100, 2));

    overallAverageK12 = np.array(averageArrK12)
    overallAverageK8 = np.array(averageArrK8)
    overallAverageK4 = np.array(averageArrK4)

    averageCollection.append(round(overallAverageK4.mean(), 2))
    averageCollection.append(round(overallAverageK8.mean(), 2))
    averageCollection.append(round(overallAverageK12.mean(), 2))

    print("\nK=4,", variation)
    print("Average Accuracy:", round(overallAverageK4.mean(), 2), averageArrK4)
    print("\nK=8,", variation)
    print("Average Accuracy:", round(overallAverageK8.mean(), 2), averageArrK8)
    print("\nK=12,", variation)
    print("Average Accuracy:", round(overallAverageK12.mean(), 2), averageArrK12)


## Bar graph for report
plt.figure(1)

classesString = ["kNN 4\nEuclidean\nDistance", "kNN 8\nEuclidean\nDistance", "kNN 12\nEuclidean\nDistance", "kNN 4\nEuclidean\nw/ z-score", "kNN 8\nEuclidean\nw/ z-score", "kNN 12\nEuclidean\nw/ z-score","kNN 4\nManhattan\nw/ z-score","kNN 8\nManhattan\nw/ z-score","kNN 12\nManhattan\nw/ z-score"]
plt.bar(classesString, averageCollection, color=['blue', 'red', 'green', 'cyan', 'pink', 'orange', 'yellow', 'purple', 'lime'])
plt.title("Average of different kNN Algorithms runs")
plt.xticks(classesString)
plt.xlabel("kNN Algorithms")
plt.ylabel("Average % of kNN ")
plt.ylim(ymin=74, ymax=90)


plt.show()