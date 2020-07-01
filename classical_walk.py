import numpy as np

A = np.array([  [0,1,0,0,1,1,0,0],
                [0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,1,0],
                [0,0,1,0,1,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,1,0,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0]])

np.nonzero(A[0,:] == 1)[0]

def cwalk(A, N):
    node = 0 # which node are we at? 0-based like indexing.
    for f in flips:
        neighbors = np.nonzero( A[0,:] == 1 )[0]
        node = neighbors[ np.random.randint(0,len(neighbors)) ]
