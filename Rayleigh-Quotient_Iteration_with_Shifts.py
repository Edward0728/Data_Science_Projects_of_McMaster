# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 22:08:47 2021

@author: forfu
"""

import numpy as np
import random
def GaussElimination (a,b):
    n=len(b)
    step=0
    #print('a,b',a,b)
    for k in range(0, n-1):
        for i in range(k+1, n):
            ratio = a[i, k]/a[k, k]
            for j in range(k, n):
                a[i, j] = a[i, j] - ratio*a[k, j]
            b[i,0] = b[i,0] - ratio*b[k,0]
            step=step+1
    
    x = np.zeros((3,1))
    
    x[n-1, 0] = b[n-1,0]/a[n-1,n-1]
    for i in range(n-2, -1, -1):
        sum_j = 0
        for j in range(i+1, n):
            sum_j = sum_j + a[i, j]*x[j]
        x[i, 0] = (b[i,0] - sum_j)/a[i, i]
    return x


A = np.array([[1., -1., 0], [0., -4., 2], [0, 0, -2]])
x0 = np.array([ [5.], [1], [1.]])
I = np.array([[1., 0, 0], [0, 1., 0],[0, 0, 1]])
for i in range(0, 3):
    x0[i,0] = random.randint(1, 10)

for i in range(0, 3):
    x0[i,0] = random.randint(1, 10)
    
errorTol = 0.01
currentError = 10*errorTol
count = 1
nrmx0_old = 100
sk = 0
#print('k, xkstandardized, Ratio/Lambda')

while currentError > errorTol:
    xold = x0
    A = np.array([[1., -1., 0], [0., -4., 2], [0, 0, -2]])
    #print('xold/x0 ', xold)
    temp1 = np.transpose(xold).dot(A).dot(xold)
    #print('temp1 ', temp1)
    temp2 = np.transpose(xold).dot(xold)
    #print('temp2 ', temp2)
    
    sk = float(temp1.dot(np.linalg.inv(temp2)))

    print('sk ', sk)
    #print('x0 \n', x0)
    #print('A-sk*I', A-sk*I)
    x0 = GaussElimination((A-sk*I), xold)
    #print('yk ', x0)
   
    nrmx0 = np.linalg.norm(x0, np.inf, None)
    #print('nrmx0 ', nrmx0)
    x0 = x0/nrmx0
    print (count, 'x0 = ', x0, 'nrmx0 = ', nrmx0,'\n\n\n')
    count = count + 1
    currentError = abs(nrmx0-nrmx0_old)/nrmx0_old*100
    nrmx0_old =  nrmx0