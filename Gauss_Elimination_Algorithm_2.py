# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 08:54:33 2021

@author: forfu
"""
import numpy as np
A = np.array([[ 1.,  2.,  1., -1.],
              [ 3.,  2.,  4.,  4.],
              [ 4,   4.,  3.,  4.],
              [ 2.,  0.,  1.,  5.]])

b = np.array([ 5., 16., 22., 15.])
n=len(b)
step=0

# Forward Elimination
# a_ij = a_ij - a_ik/a_kk * a_kj
# b_i = b_i - a_ik/a_kk * b_k
# k: pivot(diagonals) -- 0 to n-2
# i: rows -- k+1 to n-1
# j: cols -- k to n-1

for k in range(0, n-1):
    p=max(abs(A_p) for A_p in A[k:n,k])
    # print("k= ", k)
    # print("p = ", p)
    #print(A[k:n,k])
    p_i = abs(A[k:n,k]).argmax()+k
    #print("p_i = ", p_i)
    A[[k, p_i],:]=A[[p_i, k], :]
    b[k],b[p_i] = b[p_i],b[k]
    #print("pivoting A: \n", A)

    for i in range(k+1, n):
        
        
        ratio = A[i, k]/A[k, k]
        for j in range(k, n):
            A[i, j] = A[i, j] - ratio*A[k, j]
        b[i] = b[i] - ratio*b[k]
        step=step+1
        print("Pivoting step %d" %(step))
        print(A)
        print(b)
        # print("k= ", k)
        # print("p = ", p)
        # print(A[k:n,k])
        # print("p_i = ", p_i)
        print("")
# Back Substitution
# x_n-1 = b_n-1/A_n-1,n-1
# x_i = (b_i - sum_j(a_ij*x_j))/a_ii

x = np.zeros(n)

x[n-1] = b[n-1]/A[n-1,n-1]
for i in range(n-2, -1, -1):
    sum_j = 0
    for j in range(i+1, n):
        sum_j = sum_j + A[i, j]*x[j]
    x[i] = (b[i] - sum_j)/A[i, i]
print("")   
print("up triangular metrix: \n", A)
print(" ", b)
print("solution vector: \n", x)

