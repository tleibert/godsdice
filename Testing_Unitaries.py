# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 17:15:34 2020

@author: Thoma
"""

# %% codecell
# imports
import numpy as np
import math
import numpy.linalg as la
import scipy.linalg as scila
from numpy import kron

# %% codecell
# setup functions

#Adjacency standard of Aij is a line from i to j

def indegrees(M,node):
    #This is the sum of the colume at node X, counting from 0
    sumCurrent = 0
    for index in range(0,M.shape[0]):
        sumCurrent += M[index][node]
    return sumCurrent

def outdegrees(M,node):
    #Counts the entries in the node row
    return sum(M[node])

def aVal(M,x):
    indegree = indegrees(M,x)
    outdegree = outdegrees(M,x)
    if indegree == 0 and outdegree == 0:
        return 1
    else:
        return indegree / (indegree + outdegree)

def constructC(M):

    # Force input matrix to already have dimensions as a power of 2
    if not np.isclose(np.log2(M.shape[0]), int(np.log2(M.shape[0]))):
        print("Please pad your adjacency matrix so its dimensions are a power of 2")

    output = np.zeros([2 * x for x in M.shape])

    for x in range(M.shape[0]):

        xVec = np.zeros(M.shape[0])
        xVec[x] = 1
        outerX = np.diag(xVec)

        a = aVal(M,x)
        cellElement = np.array([[np.sqrt(1/(a + 1)),np.sqrt(a/(1 + a))],[np.sqrt(a/(1 + a)),-1 * np.sqrt(1/(a + 1))]])
        output += kron(cellElement,outerX)

    return output

def constructS(M):

    # Force input matrix to already have dimensions as a power of 2
    if not np.isclose(np.log2(M.shape[0]), int(np.log2(M.shape[0]))):
        print("Please pad your adjacency matrix so its dimensions are a power of 2")

    up_proj = np.diag([1,0])
    down_proj = np.diag([0,1])

    P,D,Q = la.svd(M)
    U = P.T.conj() @ scila.expm(1j*np.diag(D)) @ Q.T.conj()

    if_up = np.kron(up_proj, np.identity(M.shape[0]))
    if_down = np.kron(down_proj, U)
    S = if_up+if_down

    return S

# # %% codecell
# #For the test case:
# A = np.array([  [0,1,0,0,1,1,0,0],
#                 [0,0,0,0,0,0,0,0],
#                 [1,1,0,0,0,0,1,0],
#                 [0,0,1,0,1,1,0,0],
#                 [0,0,0,0,0,0,1,0],
#                 [0,0,0,1,0,0,0,0],
#                 [0,0,0,0,0,1,0,0],
#                 [0,0,0,0,0,0,0,0]])
# #Finding C
# C = constructC(A)
# C = np.matrix(C)
#
# S = constructS(A)
# S = np.matrix(S)
#
