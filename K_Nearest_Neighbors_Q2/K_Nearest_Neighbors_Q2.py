
import numpy as np
from numpy import *
from math import *
from os import *
import pandas as pd
from sklearn import preprocessing
#Assuming that all of the binary data can be treated as symmetric, use k nearest neighbor classification with k = 3, 5 and
#7, classify the following data for two different attributes – bladder inflammation (yes, no) and Nephritis (yes, no).
#Additionally, determine how many errors we would obtain for the two classification results for each value of k.

df = pd.read_excel(r'D:\Google Drive 1\McMaster Degree\Level 4-2\4DM3\week_2\SFWRTECH 4DM3Data Mining - 1122022 - 647 PM\Assignment_2_data.xlsx', sheet_name='Q2')
test= pd.read_excel(r'D:\Google Drive 1\McMaster Degree\Level 4-2\4DM3\week_2\SFWRTECH 4DM3Data Mining - 1122022 - 647 PM\Assignment_2_data.xlsx', sheet_name='Test2')

df_labeled=df.copy()
test_original=test.copy()
#df.pop('Bladder Inflamation')
df.pop('Nephritis')
#test.pop('Bladder Inflamation')
test.pop('Nephritis')

#label encoder for nominal data
le = preprocessing.LabelEncoder()
#cols = ['Nausea','Lumbar Pain','Urine Pushing','Micturition Pain','Burning on Urination','Nephritis']
cols = ['Nausea','Lumbar Pain','Urine Pushing','Micturition Pain','Burning on Urination','Bladder Inflamation']
df[cols] = df[cols][1:].apply(le.fit_transform)
NumData = df.to_numpy()
NumData = NumData[1:,1:]

le = preprocessing.LabelEncoder()
test[cols] = test[cols][1:].apply(le.fit_transform)
TestData = test.to_numpy()
TestData = TestData[1:-1,1:]

#concatenate data and test record
NumData = np.insert(NumData,0,TestData, axis = 0)

# numeric normalization
def Numeric_norm(NumericData):
    Numeric_norm_matrix = zeros(NumericData.shape)
    for i in range(len(NumData)):
        Numeric_norm_matrix[i] = (NumericData[i]-min(NumericData[:]))/(max(NumericData[:])-min(NumericData[:]))
    return Numeric_norm_matrix

Numeric_norm_matrix = Numeric_norm(NumData[:,0])

#calculate Manhattan distance of numeric data 
def Man_numeric_diff(NumericData,test):
    Man_numeric_matrix = zeros((len(test),len(NumericData)))
    for i in range(len(test)):
        for j in range(len(NumericData)):
            Man_numeric_matrix[i,j] = abs(test[i] - NumericData[j])
    return Man_numeric_matrix

Man_numeric_matrix = Man_numeric_diff(Numeric_norm_matrix[len(TestData):],Numeric_norm_matrix[0:len(TestData)])


#calculate Mahattan distance of nominal data
def Man_nominal_diff(NominalData,test):
    Man_nominal_matrix = zeros((len(test),len(NominalData)))
    for i in range(len(test)):
        for j in range(len(NominalData)):
            for k in range(NominalData.shape[1]):
                Man_nominal_matrix[i,j] += np.where(test[i,k]==NominalData[j,k],0,1)/(NominalData.shape[1])
    return Man_nominal_matrix

Man_nominal_matrix = Man_nominal_diff(NumData[len(TestData):,1:NumData.shape[1]],NumData[0:len(TestData),1:NumData.shape[1]])


#calculate Mahattan distance of all data
def Man_all_diff(Nom_diff,Num_diff,test,nom_cols):
    Man_all_diff = zeros((len(test),Nom_diff.shape[1]))
    for i in range(len(test)):
        for j in range(Nom_diff.shape[1]):
            Man_all_diff[i,j] = (Nom_diff[i,j]*len(nom_cols) + Num_diff[i,j]*(TestData.shape[1]-len(nom_cols))) /7  #number of attributes 
    return Man_all_diff

Man_all_matrix = Man_all_diff(Man_nominal_matrix,Man_numeric_matrix,TestData,cols)

#assemble the distance with original dataset
def assemble_data(Dist_matrix, data):
    for i in range(len(Dist_matrix)):
        Assemble_matrix = np.concatenate((data,Dist_matrix.T), axis = 1)
    return Assemble_matrix

Data = df_labeled.to_numpy()
Data = Data[1:,:]
Assemble_matrix = assemble_data(Man_all_matrix, Data)


# Locate the most similar neighbors
def get_neighbors(Assemble_matrix, test_record_number,num_neighbors):
    Assemble_matrix_sorted = Assemble_matrix[np.argsort(Assemble_matrix[:,8+test_record_number])] #number of columns in original data
    neighbors = list()
    for i in range(num_neighbors):
	    neighbors.append(Assemble_matrix_sorted[i])
    return neighbors

######## Please customize your prediction here !!! #######
test_record_number = 1
num_neighbors = 7
neighbors = get_neighbors(Assemble_matrix,test_record_number,num_neighbors)


# Make a classification prediction with neighbors
def predict_classification(neighbors):
	output_values = [row[8] for row in neighbors] #target column number
	prediction = max(set(output_values), key=output_values.count)
	return prediction

pred = predict_classification(neighbors)

print(test_original.loc[[test_record_number]], pred)

### Conclustion (k=7):

## For Bladder Inflamation:
# For patient 4: the prediction is yes

## For Nephritis: 
# prediction on patient 1, the 7 neighbors are {2,1,6,13,11,12,17} instead of {2,1,6,13,7,5,8} given in solution
# All predictions match the given solutions.