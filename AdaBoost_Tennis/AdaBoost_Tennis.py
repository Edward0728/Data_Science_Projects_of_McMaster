from numpy import *
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import *
from matplotlib.pyplot import *
from os import *
print('reading data')
# read the training and test files
rawData = loadtxt(r'D:\Google Drive 1\McMaster Degree\Level 4-2\4DM3\week_6\tennis.txt', delimiter = ',')
print('Done')
# separate the training data and the class
trainingData = rawData[:100,1:]
trainingDataClass = rawData[:100,0]
# separate the test data from the class
testData = rawData[100:,1:]
testDataClass = rawData[100:,0]
trainedClassifier = AdaBoostClassifier(n_estimators = 40, random_state=0)
trainedClassifier.fit(trainingData, trainingDataClass)
prediction = trainedClassifier.predict(testData)
numberCorrect = sum((testDataClass == prediction.T).astype(int))
errorRate = 1. - (numberCorrect / testDataClass.size)
print('error rate = ')
print(errorRate)
