import numpy as np

def connected_node_mask(A):
    node_mask = [] # true for connected nodes, false for disconnected nodes
    for i in range(A.shape[0]):
        if False not in np.equal( A[i,:], np.zeros(A.shape[0]) ) and False not in np.equal( A[:,i], np.zeros(A.shape[0]) ):
            node_mask.append(False)
        else:
            node_mask.append(True)
    return node_mask

def cwalk(A, N):
    connected_nodes = A[connected_node_mask(A)]
    counts = np.zeros( len(connected_nodes) )

    node = 0 # which node are we at? 0-based like indexing.
    for i in range(N):
        counts[node] += 1

        # calculate the new node that we walk to
        neighbors = np.nonzero( A[0,:] == 1 )[0]
        node = neighbors[ np.random.randint(0,len(neighbors)) ]

    ranks = counts / np.sum(counts)
    return ranks
