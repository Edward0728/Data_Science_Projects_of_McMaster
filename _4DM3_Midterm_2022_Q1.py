import numpy as np
import random
from numpy import *


mean = [10,15,20]
variance = [3,4,5]
size = 100


Num_data=[]
for i in range(3):
    attribute = np.random.normal(loc=mean[i], scale=variance[i]**0.5, size=size)
    Scale_attri = zeros((size,))
    for j in range(size):
        Scale_attri[j]=abs(attribute[j]-min(attribute))/(max(attribute)-min(attribute))
    Num_data.append(Scale_attri)
Num_data=np.array(Num_data).T



Nom_data=[]
random.seed =(1)
for i in range(3):
    attribute = random.random_integers(low=1, high=5, size=size)
    Scale_attri = zeros((size,))
    for j in range(size):
        Scale_attri[j]=abs(attribute[j]-min(attribute))/(max(attribute)-min(attribute))
    Nom_data.append(Scale_attri)
Nom_data=np.array(Nom_data).T


Ord_data=[]
Ord_data_noscale = []
for i in range(3):
    attribute = random.random_integers(low=1, high=10, size=size)
    Ord_data_noscale.append(attribute)
    Scale_attri = zeros((size,))
    for j in range(size):
        Scale_attri[j]=abs(attribute[j]-min(attribute))/(max(attribute)-min(attribute))
    Ord_data.append(Scale_attri)
Ord_data=np.array(Ord_data).T
Ord_data_noscale = np.array(Ord_data_noscale).T




#(a) Euclidean Distance of numeric attributes
DNumE = mat(zeros((size,size)))

for i in range(size): # number of rows
    for j in range(size): # number of rows
        for k in range(3): # number of values in every row
            DNumE[i,j] = DNumE[i,j] + (Num_data[i,k]-Num_data[j,k])**2.0 # Euclidian formular
        DNumE[i,j] = (DNumE[i,j])**0.5
        

#(b) Manhattan Distance of numeric attributes
DNumM = mat(zeros((size,size)))

for i in range(size): # number of rows
    for j in range(size): # number of rows
        for k in range(3): # number of values in every row
            DNumM[i,j] = DNumM[i,j] + abs(Num_data[i,k]-Num_data[j,k])  # Manhattan formular


#(c) Supremum Distance of numeric attributes
DNumMax = mat(zeros((size,size)))
DMax = mat(zeros((3,1)))
for i in range(size): # number of rows
    for j in range(size): # number of rows
        for k in range(3): # number of values in every row
            DMax[k] = abs(Num_data[i,k]-Num_data[j,k])  # Supremum formular
        DNumMax[i,j] = max(DMax)

#(d) Dissmilarity Matrix of nominal
DNom = mat(zeros((size,size)))
for i in range(size):
    for j in range(size):
        for k in range(3):
            DNom[i,j] = DNom[i,j] + int(Nom_data[i,k]==Nom_data[j,k]) # number of same attributes
        DNom[i,j] = (3-DNom[i,j])/3.0; # d(i,j)= (p-m)/p


#(e) Dissmilarity Matrix of ordinal, use Euclidean Distance
DOrdE = mat(zeros((size,size)))
DOrdE_noscale = mat(zeros((size,size)))
for i in range(size): # number of rows
    for j in range(size): # number of rows
        for k in range(3): # number of values in every row
            DOrdE_noscale[i,j] = DOrdE_noscale[i,j] + (Ord_data_noscale[i,k]-Ord_data_noscale[j,k])**2.0 # Euclidian formular
            DOrdE[i,j] = DOrdE[i,j] + (Ord_data[i,k]-Ord_data[j,k])**2.0 # Euclidian formular
        DOrdE[i,j] = (DOrdE[i,j])**0.5
        DOrdE_noscale[i,j] = (DOrdE_noscale[i,j])**0.5


#(f) Overall Dissimilarity, use Euclidean for numeric and ordianl

DOverall = (3.0*DNumE + 3.0*DOrdE + 3.0*DNom)/9.0


aaa