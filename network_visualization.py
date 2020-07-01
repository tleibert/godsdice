import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

import pageRankSim

def connected_node_mask(A):
    node_mask = [] # true for connected nodes, false for disconnected nodes
    for i in range(A.shape[0]):
        if False not in np.equal( A[i,:], np.zeros(A.shape[0]) ) and False not in np.equal( A[:,i], np.zeros(A.shape[0]) ):
            node_mask.append(False)
        else:
            node_mask.append(True)
    return node_mask

# fname should be .mp4!
def network_viz(A, rankmatrix, fps, fname):
    node_mask = connected_node_mask(A)

    G = nx.from_numpy_matrix( A[node_mask][:,node_mask] )

    fig, ax = plt.subplots()

    pos=nx.circular_layout(G)

    def update_nx(row, data):
        ax.cla()
        #print(pos)
        for k in pos.keys():
            #print(k)
            x = pos[k][0]
            y = pos[k][1]
            circ = plt.Circle( ( x, y ), 0.095, fill=False, edgecolor="black" )
            ax.add_artist(circ)

            ax.text( x + 0.05, y + 0.05, int(k) + 1 )
        ax.axis("equal")
        nx.draw_networkx(G, pos=pos, ax=ax, cmap="gray", node_color=data[row,:], with_labels=False)
        ax.axis('off')

    update_nx(0, rankmatrix) # initialization

    sm = plt.cm.ScalarMappable(cmap="gray", norm=plt.Normalize(vmin=np.min(rankmatrix), vmax=np.max(rankmatrix)))
    plt.colorbar(sm, ax=ax)

    anim = animation.FuncAnimation(fig, update_nx, rankmatrix.shape[0], fargs=(rankmatrix, ))

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps)
    anim.save(fname, writer=writer)

# %% codecell
    
    #Example animation:
    
A = np.array([  [0,1,0,0,1,1,0,0], #Defining adjacency matrix
                 [0,0,0,0,0,0,0,0],
                 [1,1,0,0,0,0,1,0],
                 [0,0,1,0,1,1,0,0],
                 [0,0,0,0,0,0,1,0],
                 [0,0,0,1,0,0,0,0],
                 [0,0,0,0,0,1,0,0],
                 [0,0,0,0,0,0,0,0]])

A=(A+A.T)>0 #Making sure A is symmetric (this makes it a non-directed graph)

rmat = pageRankSim.beginsim(A,50) #creating a rankmatrix with graph A, 50 animation frames

fps = 2 #setting fps

fname = "7nodenondirect.mp4" #naming animation output file

network_viz(A, rmat, fps, fname) #Calling visualization function to make the animation
