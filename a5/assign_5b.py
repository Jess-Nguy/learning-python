"""
MLP does beter than linear regression because the data set is scattered which fits the many hidden layers I set for MLP.

https://archive.ics.uci.edu/ml/datasets/Behavior+of+the+urban+traffic+of+the+city+of+Sao+Paulo+in+Brazil

Jess Nguyen
"""

import numpy as np
import csv
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

data = []
targets = []
# Read csv file
with open("traffic.csv") as file:
    for line in file:
        row = line.strip().split(",")
        frow = []
        for item in row:
            frow += [float(item)]
        data += [frow[:-1]]
        targets += [frow[-1]]
data = np.array(data)
targets = np.array(targets)

# Split data
dataLength = round(len(data) * 0.75)

traindata = np.array(data[:dataLength])
traintargets = np.array(targets[:dataLength])

testdata = np.array(data[dataLength:])
testtargets = np.array(targets[dataLength:])

print("----TRAFFIC----")
print("Training length:", dataLength)
print("Testing length:", len(data)-dataLength)
print("Number of Features:", np.size(data, 1))

# Create and train linear regressor
rgr = LinearRegression()
rgr.fit(traindata, traintargets)

# Print the coefficients and intercepts of the model
print("\nCoefficients:", rgr.coef_)
print("Intercept:", round(rgr.intercept_, 2))

# Test the model and report the result
#  Note that sklearn only lets you get R^2. You can take the square
#  root to get the correlation coefficient, but it will always come
#  out positive. So use the np.corrcoef function as shown below instead - CODE BY: SAM SCOTT
pred = rgr.predict(testdata)
print("Correlation (r):", round(np.corrcoef(pred, testtargets)[0, 1], 2))
print("Residual Sum of Squares:", round(((pred-testtargets)**2).sum(), 2))

# Normalize Training data and target
# traindata = (traindata-traindata.min(axis=0))/(traindata.max(axis=0)-traindata.min(axis=0))
#
# traintargets = (traintargets-traintargets.min(axis=0))/(traintargets.max(axis=0)-traintargets.min(axis=0))

# Create and train MLP regressor
mlp = MLPRegressor(learning_rate_init=0.0005,
                   hidden_layer_sizes=[100, 50, 25, 12, 6])
mlp.fit(traindata, traintargets)

# Test the model and report the result MLP
pred = mlp.predict(testdata)
print("\nMLP Correlation (r):", round(np.corrcoef(pred, testtargets)[0, 1], 2))
print("MLP Residual Sum of Squares:", round(((pred-testtargets)**2).sum(), 2))
