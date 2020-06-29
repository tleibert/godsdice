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


from qiskit.visualization import plot_histogram

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
    U = P @ scila.expm(1j*np.diag(D)) @ Q
    if_up = np.kron(up_proj, np.identity(M.shape[0]))
    if_down = np.kron(down_proj, U)
    S = if_up+if_down

    return S

def returnU(M):
    # Force input matrix to already have dimensions as a power of 2
    if not np.isclose(np.log2(M.shape[0]), int(np.log2(M.shape[0]))):
        print("Please pad your adjacency matrix so its dimensions are a power of 2")

    up_proj = np.diag([1,0])
    down_proj = np.diag([0,1])

    P,D,Q = la.svd(M)
    U = P @ scila.expm(1j*np.diag(D)) @ Q
    return U

def manualSVD(M):
    leftMatrix = M @ M.conjugate()
    rightMatrix = M.conjugate() @ M
    rightEigVal, rightEigVec = scila.eig(rightMatrix,None, False, True)
    # is this right? with M instead of leftMatrix
    leftEigVal, leftEigVec = scila.eig(M,None, True, False)
    return(leftEigVec,leftEigVal,rightEigVal,rightEigVec)

# %% codecell
# #For the test case:
A = np.array([  [0,1,0,0,1,1,1,0],
                [0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,1,0],
                [0,0,1,0,1,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,0]])

# initialize statevector, operators
C = constructC(A)
S = constructS(A)
SC = S @ C

initialState = np.zeros(8)
active_nodes = 7

for i in range(active_nodes):
    initialState[i] = 1/np.sqrt(active_nodes)

psio = kron(np.array([1,1]),initialState)/np.sqrt(2)

# %% codecell
# run the circuit - get SV after one step, then take one more step, etc.
finalStates = [SC@psio]
for i in range(100):
    finalStates.append(SC @ finalStates[i])

# find probabilities (instantaneous rank) after each step
probabilityList = []
for item in range(len(finalStates)):
    probabilities = [np.real(p * np.conj(p)) for p in finalStates[item]]
    probabilityList.append(probabilities)

# combine instantaneous ranks into final average rank
counts = {'000':0,'001':0,'010':0,'011':0,'100':0,'101':0,'110':0,'111':0}
for item in probabilityList:
    for i in range(len(item)):
        # figures out the binary label that qiskit would return
        binI = bin(i)[2:].zfill(4)[1:]
        counts[binI] += item[i]

plot_histogram(counts)

# %% codecell

# #Finding C
# C = constructC(A)
# C = np.matrix(C)
#
# S = constructS(A)
# S = np.matrix(S)
#
"""
Created on Sat Jun 27 17:15:34 2020

@author: Thoma


import numpy as np
import math
import numpy.linalg as la
import scipy.linalg as scila
from numpy import kron

#Adjaceny standard of Aij is a line from i to j
#kronBuilder can be fixed

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
    output = np.zeros([2 * x for x in M.shape])

    for x in range(M.shape[0]):
        xVec = np.zeros(M.shape[0])
        xVec[x] = 1
        outerX = np.outer(xVec,xVec)
        a = aVal(M,x)
        cellElement = np.array([[np.sqrt(1/(a + 1)),np.sqrt(a/(1 + a))],[np.sqrt(a/(1 + a)),-1 * np.sqrt(1/(a + 1))]])
        output += kron(cellElement,outerX)
    return output


#WIP
def constructS(M,U):
    dimensions = math.ceil(np.log2(M.shape[0]))
    output = np.zeros([2 * x for x in M.shape],dtype=np.complex128)
    up = np.array([1,0])
    down = np.array([0,1])

    for x in range(M.shape[0]):
        xVec = np.zeros(M.shape[0])
        xVec[x] = 1
        outerX = np.outer(xVec, xVec)
        firstTerm = kron(np.outer(up,up),outerX)
        secondTermOutput = np.zeros([2 * x for x in M.shape],dtype=np.complex128)
        for k in range(M.shape[0]):
            kVec = np.zeros(M.shape[0])
            kVec[k] = 1
            kOuterX = np.outer(kVec,xVec)
            outerDown = np.outer(down,down)
            #CHECK THESE COORDS
            secondTermOutput += kron(outerDown,U[k][x] * kOuterX)
        output += firstTerm
        output += secondTermOutput
    return output


#For the test case:
#A = np.array([[0,1,0,0,1,1,1,0],[0,0,0,0,0,0,0,0],[1,1,0,0,0,0,1,0],[0,0,1,0,1,1,0,0],[0,0,0,0,0,0,1,0],[0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0]])
#PDQ = la.svd(A)
#P = PDQ[0]
#D = np.diag(PDQ[1])
#Q = PDQ[2]
#U = P @ scila.expm(1j*D) @ Q


#print(indegrees(A,1))
#print(outdegrees(A,1))
#print(indegrees(A,0))
#print(outdegrees(A,0))

#Finding C
#C = constructC(A)
#C = np.matrix(C)
#print(np.round(C.H @ C,3))
#print(C)
#S = constructS(A,U)
#S = np.matrix(S)
#print(np.round(S.H @ S,3).real)
'''
print(la.det(constructC(A)))
print(la.det(constructS(A,U)))
U = np.matrix(U)
print(np.round(U.H @ U,3))
print(la.det(U))

'''



#xTest = kronBuilder('111',3)
#print(np.zeros([2 * x for x in A.shape]))
#print(np.outer(xTest))

#print(kronBuilder('111',3))
#print(math.ceil(np.log2(A.shape[0])))

"""
