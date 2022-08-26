"""
Jess Nguyen

Reading csv of phising data set to summarize, plot and bar graph the data.
https://archive.ics.uci.edu/ml/datasets/Website+Phishing

September 28 2021
"""

import csv
import numpy as np
from matplotlib import pyplot as plt

phishingData = []
labels = []

# Read csv file
with open("./../PhishingData.csv") as file:
    reader = csv.DictReader(file, delimiter=",")
    # loop through lines and append to arrays by column names
    for row in reader:
        labels.append(int(row["Result"]))
        dataRow = [int(row["SFH"]), int(row["popUpWindow"]), int(row["SSLfinal_State"]), int(row["Request_URL"]), int(
            row["URL_of_Anchor"]), int(row["web_traffic"]), int(row["URL_Length"]), int(row["age_of_domain"]), int(row["having_IP_Address"])]
        phishingData.append(dataRow)
    featureNames = list(row.keys())

# Make lists into numpy array
featureNames = np.array(featureNames[1:10])
labels = np.array(labels)
phishingData = np.array(phishingData)

# Shuffling the phishingData & labels
arangeShuffle = np.arange(phishingData.shape[0])
np.random.shuffle(arangeShuffle)
phishingData = phishingData[arangeShuffle]
labels = labels[arangeShuffle]

# Calculate how many rows is 75% and how many rows is 25%
totalRows = len(phishingData)
trainingNumRows = round(totalRows * 0.75)
testingNumRows = totalRows - trainingNumRows

# Get the first 75% of the phishingData to be training
trainingData = np.array(phishingData[:trainingNumRows])
trainingLabels = np.array(labels[:trainingNumRows])

# Get the last 25% of the phishingData to be testing
testingData = np.array(phishingData[testingNumRows:])
testingLabels = np.array(labels[testingNumRows:])


# names of the classes
names = sorted(set(labels))

# Summary

print("TRAINING SET ")
print("Names: ", list(featureNames))
print("Minima: ", list(trainingData.min(axis=0)))
print("Maxima: ", list(trainingData.max(axis=0)))
print("Mean: ", list(np.round(trainingData.mean(axis=0), 2)))
print("Median: ", list(np.median(trainingData, axis=0)))

print("TESTING SET")
print("Names: ", list(featureNames))
print("Minima: ", list(testingData.min(axis=0)))
print("Maxima: ", list(testingData.max(axis=0)))
print("Mean: ", list(np.round(testingData.mean(axis=0), 2)))
print("Median: ", list(np.median(testingData, axis=0)))

# Scatter
# plot data by training labels

plt.figure(1)

plt.scatter(trainingData[trainingLabels == -1][:, 0],
            trainingData[trainingLabels == -1][:, 1], c="blue", marker=".")
plt.scatter(trainingData[trainingLabels == 0][:, 0],
            trainingData[trainingLabels == 0][:, 1], c="red", marker="*")
plt.scatter(trainingData[trainingLabels == 1][:, 0],
            trainingData[trainingLabels == 1][:, 1], c="green", marker="^")

plt.title("Phishing Data for " + featureNames[0] + " and " + featureNames[1])
plt.xlabel(featureNames[0])
plt.ylabel(featureNames[1])


plt.figure(2)

plt.scatter(trainingData[trainingLabels == -1][:, 2],
            trainingData[trainingLabels == -1][:, 3], c="blue", marker=".")
plt.scatter(trainingData[trainingLabels == 0][:, 2],
            trainingData[trainingLabels == 0][:, 3], c="red", marker="*")
plt.scatter(trainingData[trainingLabels == 1][:, 2],
            trainingData[trainingLabels == 1][:, 3], c="green", marker="^")

plt.title("Phishing Data for " + featureNames[2] + " and " + featureNames[3])
plt.xlabel(featureNames[2])
plt.ylabel(featureNames[3])


plt.figure(3)

plt.scatter(trainingData[trainingLabels == -1][:, 4],
            trainingData[trainingLabels == -1][:, 5], c="blue", marker=".")
plt.scatter(trainingData[trainingLabels == 0][:, 4],
            trainingData[trainingLabels == 0][:, 5], c="red", marker="*")
plt.scatter(trainingData[trainingLabels == 1][:, 4],
            trainingData[trainingLabels == 1][:, 5], c="green", marker="^")

plt.title("Phishing Data for " + featureNames[4] + " and " + featureNames[5])
plt.xlabel(featureNames[4])
plt.ylabel(featureNames[5])


# Bar graph
# bar graph the data by training labels

plt.figure(4)

sumOfData = [len(trainingData[np.where(trainingLabels == -1)]), len(
    trainingData[np.where(trainingLabels == 0)]), len(trainingData[np.where(trainingLabels == 1)])]

classesString = ['Phishy -1', 'Suspicious 0', 'Legitimate 1']
plt.bar(classesString, sumOfData, color=['blue', 'red', 'green'])
plt.title("Phishing Data for # of class occurrence")
plt.xticks(classesString)
color = ['black', 'red', 'green', 'blue', 'cyan']
plt.xlabel("Classification")
plt.ylabel("Total Number of Occurrence")

plt.show()
