"""
Jess Nguyen

Reading csv of phising data set to run training and testing predictions against the dataset with sklearn's decision trees. Tested out best parameters and values for the decision trees.
https://archive.ics.uci.edu/ml/datasets/Website+Phishing

October 17 2021
"""

import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree

phishingData = []
labels = []

## Read csv file
with open("PhishingData.csv") as file:
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


def averageLoop(clf, phishingData, labels):
    """
    Run displayPrediction in a loop for each different parameter and value.
    clf = DecisionTreeClassifier, classifier parameter and values for Decision tree.
    phisingData = 2d array, the whole phising data set.
    labels = 1d array, classification of the training data set.
    """
    accuracyArray = []
    for runs in range(0,5):
        accuracyArray.append(displayPrediction(clf, phishingData, labels))

    overallAccuracy = np.array(accuracyArray)
    print("Average Accuracy:", round(overallAccuracy.mean(), 2), accuracyArray)


def displayPrediction(clf, phishingData, labels):
    """
    Run training fit then testing prediction.
    clf = DecisionTreeClassifier, classifier parameter and values for Decision tree.
    phisingData = 2d array, the whole phising data set.
    labels = 1d array, classification of the training data set.
    """
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

    clf = clf.fit(trainingData, trainingLabels)

    prediction = clf.predict(testingData)
    correct = (prediction == testingLabels).sum()
    return round(correct/len(prediction)*100,2)


# Max Depth
valueTest = 4
print("\n\n====== Max Depth:", str(valueTest), ", ENTROPY, Min Sample Leaf: DEFAULT, Max Leaf Nodes: DEFAULT======")
clf = tree.DecisionTreeClassifier(max_depth=4, criterion="entropy")
averageLoop(clf, phishingData, labels)

valueTest = 20
print("\n\n====== Max Depth", str(valueTest), ", GINI, Min Sample Leaf: DEFAULT, Max Leaf Nodes: DEFAULT======")
clf = tree.DecisionTreeClassifier(max_depth=20, criterion="gini")
averageLoop(clf, phishingData, labels)

# Criterion

print("\n\n====== GINI, Max Depth: DEFAULT, Min Sample Leaf: DEFAULT, Max Leaf Nodes: DEFAULT ======")
clf = tree.DecisionTreeClassifier(criterion="gini")
averageLoop(clf, phishingData, labels)

print("\n\n====== ENTROPY, Max Depth: DEFAULT, Min Sample Leaf: DEFAULT, Max Leaf Nodes: DEFAULT ======")
clf = tree.DecisionTreeClassifier(criterion="entropy")
averageLoop(clf, phishingData, labels)


## Min sample leaf
# valueTest = 5
#
# print("\n\n====== Min Sample Leaf:", str(valueTest), ", ENTROPY, Max Depth: DEFAULT, Max Leaf Nodes: DEFAULT ======")
# clf = tree.DecisionTreeClassifier(min_samples_leaf=valueTest, criterion="entropy")
# averageLoop(clf, phishingData, labels)

valueTest = 40

print("\n\n====== Min Sample Leaf:", str(valueTest), "GINI, Max Depth: DEFAULT, Max Leaf Nodes: DEFAULT ======")
clf = tree.DecisionTreeClassifier(min_samples_leaf=valueTest, criterion="gini")
averageLoop(clf, phishingData, labels)


# Max leaf nodes
# valueTest = 4
#
# print("\n\n====== Max Leaf Nodes:", str(valueTest), ", ENTROPY, Max Depth: DEFAULT, Min Sample Leaf: DEFAULT ======")
# clf = tree.DecisionTreeClassifier(max_leaf_nodes=valueTest, criterion="entropy")
# averageLoop(clf, phishingData, labels)
#
# valueTest = 40
#
# print("\n\n====== Max Leaf Nodes:", str(valueTest), ", GINI, Max Depth: DEFAULT, Min Sample Leaf: DEFAULT ======")
# clf = tree.DecisionTreeClassifier(max_leaf_nodes=valueTest, criterion="gini")
# averageLoop(clf, phishingData, labels)


##  Graph the Decision Tree

classesString = ['Phishy -1','Suspicious 0','Legitimate 1']

import graphviz
dot_data = tree.export_graphviz(clf,
    out_file=None,
    feature_names=featureNames,
    class_names= classesString,
    filled=True,
    rounded=True,
    special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("tree.dot")
