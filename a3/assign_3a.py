"""
Jess Nguyen

Reading csv of phising data set to run training and testing predictions against the dataset with sklearn's decision trees. Tested out best parameters and values for the decision trees.
https://archive.ics.uci.edu/ml/datasets/Website+Phishing

October 26 2021
"""
import csv
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
import numpy as np

phishingData = []
labels = []

# Read csv file
with open("PhishingData.csv") as file:
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

# Calculate how many rows is 75% and how many rows is 25%
totalRows = len(phishingData)
trainingNumRows = round(totalRows * 0.75)

predictions = []

for run in range(0, 50):

    # Shuffling the phishingData & labels
    arangeShuffle = np.arange(phishingData.shape[0])
    np.random.shuffle(arangeShuffle)
    phishingData = phishingData[arangeShuffle]
    labels = labels[arangeShuffle]

    # Get the first 75% of the phishingData to be training
    trainingData = np.array(phishingData[:trainingNumRows])
    trainingLabels = np.array(labels[:trainingNumRows])

    # Get the last 25% of the phishingData to be testing
    testingData = np.array(phishingData[trainingNumRows:])
    testingLabels = np.array(labels[trainingNumRows:])

# Correct and incorrect prediction
    gnb = GaussianNB()
    gnb.fit(trainingData, trainingLabels)
    prediction = gnb.predict(testingData)
    correct = (prediction == testingLabels)
    incorrect = (prediction != testingLabels)
    accuracy = round(correct.sum()/len(prediction)*100, 2)
    predictions.append(accuracy)

correctPred = testingData[correct]
incorrectPred = testingData[incorrect]

# Average Accuracy
print(round(np.array(predictions).mean(), 2), predictions)

probsCorrect = np.array(gnb.predict_proba(correctPred).max(axis=1))
probsIncorrect = np.array(gnb.predict_proba(incorrectPred).max(axis=1))

print("Average Prediction Corrects:", round(
    np.array(probsCorrect).mean()*100, 2))
print("Average Prediction Incorrects:", round(
    np.array(probsIncorrect).mean()*100, 2))
