# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 11:38:12 2021

@author: forfu
"""

import math
import random
import sympy as sym
import numpy as np

def GaussElimination (a,b):
    n=len(b)
    step=0
    for k in range(0, n-1):
        for i in range(k+1, n):
            ratio = a[i, k]/a[k, k]
            for j in range(k, n):
                a[i, j] = a[i, j] - ratio*a[k, j]
            b[i,0] = b[i,0] - ratio*b[k,0]
            step=step+1
    
    x = np.zeros((2,1))
    
    x[n-1, 0] = b[n-1]/a[n-1,n-1]
    for i in range(n-2, -1, -1):
        sum_j = 0
        for j in range(i+1, n):
            sum_j = sum_j + a[i, j]*x[j]
        x[i, 0] = (b[i] - sum_j)/a[i, i]
    return x

x0 =np.array([[1],[2]])
A = np.array([x0[0]+2*x0[1]-2, x0[0]**2+4*x0[1]**2-4])
B = np.array([[1, 2], [2, 16]])
errorTol = 0.001
currentError = 10*errorTol
count = 1
nrmx0_old = 100
x0_origin = np.array([[0.], [0.]])


while currentError > errorTol:
    xold = x0
    Aold = A.copy()
    Bold = B.copy()
    Anegative = np.array([-(x0[0]+2*x0[1]-2),-(x0[0]**2+4*x0[1]**2-4)])
    s0 = GaussElimination(B, Anegative)
    B = Bold
    x0 = xold + s0
    A = np.array([x0[0]+2*x0[1]-2,x0[0]**2+4*x0[1]**2-4])
    y0 = A - Aold
    temp1 = ((y0 - Bold.dot(s0)).dot(np.transpose(s0)))
    temp2 = np.transpose(s0).dot(s0)
    Bx = temp1/temp2

    #print('Bx: ', Bx)
    B = Bold + Bx
    print('B: ', B)
    print('X: ', x0)
    print('f(x): ', y0)
    print(' ')
    
    x0_origin = x0.copy()
    nrmx0 = np.linalg.norm(x0, np.inf, None)
    x0 = x0/nrmx0
    count = count + 1
    currentError = abs(nrmx0-nrmx0_old)/nrmx0_old*100
    nrmx0_old =  nrmx0
    x0 = x0_origin





