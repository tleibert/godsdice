# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 17:15:34 2020

@author: Thoma
"""

import numpy as np
import math
import numpy.linalg as la
import scipy.linalg as scila
from numpy import kron

#Adjaceny standard of Aij is a line from i to j


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
        return -1
    else:
        return indegree / (indegree + outdegree)

def kronBuilder(stringElement,length):
    stringElement = stringElement.zfill(length)
    kronReturn = 1
    kronIndex = [[1,0],[0,1]]
    for item in range(length):
        kronReturn = kron(kronIndex[int(stringElement[item])],kronReturn)
    return kronReturn
    #print(kronReturn)
    
def constructC(M):
    dimensions = math.ceil(np.log2(M.shape[0]))
    output = np.zeros([2 * 2 ** dimensions, 2 * 2 ** dimensions])
    
    for x in range(M.shape[0]):
        #print(x)
        xbin = bin(x)[2:]
        #print(xbin)
        xVec = kronBuilder(xbin,dimensions)
        outerX = np.outer(xVec,xVec)
        a = aVal(M,x)
        if a != -1:
            cellElement = np.array([[np.sqrt(1/(a + 1)),np.sqrt(a/(1 + a))],[np.sqrt(a/(1 + a)),-1 * np.sqrt(1/(a + 1))]])
            output += kron(cellElement,outerX)
    return output
#WIP
def constructS(M,U):
    dimensions = math.ceil(np.log2(M.shape[0]))
    output = np.zeros([2 * 2 ** dimensions, 2 * 2 ** dimensions],dtype=np.complex128)
    up = np.array([1,0])
    down = np.array([0,1])
    
    for x in range(M.shape[0]):
        xbin = bin(x)[2:]
        xVec = kronBuilder(xbin,math.ceil(np.log2(M.shape[0])))
        outerX = np.outer(xVec, xVec)
        firstTerm = kron(np.outer(up,up),outerX)
        for k in range(M.shape[0]):
            secondTermOutput = np.zeros([2 * 2 ** dimensions, 2 * 2 ** dimensions],dtype=np.complex128)
            kbin = bin(k)[2:]
            kVec = kronBuilder(kbin,dimensions)
            kOuterX = np.outer(kVec,xVec)
            outerDown = np.outer(down,down)
            #CHECK THESE COORDS
            secondTermOutput += kron(outerDown,U[k][x] * kOuterX)
        output += firstTerm 
        output += secondTermOutput
    return output
            
        
#For the test case:
A = np.array([[0,1,0,0,1,1,0,0],[0,0,0,0,0,0,0,0],[1,1,0,0,0,0,1,0],[0,0,1,0,1,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0]])
PDQ = la.svd(A)
P = PDQ[0]
D = np.diag(PDQ[1]) 
Q = PDQ[2]
U = P @ scila.expm(1j*D) @ Q

#Finding C
print(constructC(A))
print(constructS(A,U))
print(U)
    



        
xTest = kronBuilder('111',3)
#print(np.zeros([2 * x for x in A.shape]))
#print(np.outer(xTest))

#print(kronBuilder('111',3))
#print(math.ceil(np.log2(A.shape[0])))
