# -*- coding: utf-8 -*-

# %% codecell
# initialize
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.extensions import UnitaryGate
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import Testing_Unitaries

import matplotlib.animation as animation
######################
# MAKE SURE Testing_Unitaries IS UP TO DATE!
######################

# %% codecell
# create the operators

#initialize run conditions
n_runs = 50
shots = 500
# for graph node 8 disconnected : index = 7
disc_indices = [7]

#Adjacency Matrix
A = np.array([  [0,1,0,0,1,1,0,0],
                [0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,1,0],
                [0,0,1,0,1,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,1,0,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0]])
initsize = A.shape[1]*2
init_vector = np.ones(initsize)/np.sqrt(initsize-2*len(disc_indices))

# initialize to superposition without |up>|111> or |down>|111>

init_vector[7] = init_vector[15] = 0


# %% codecell

rankmatrix = A

fig, ax = plt.subplots()

# animation
def update_hist(row, data):
    ax.clear()
    ### This will fix the y-axis; can adjust the upperbound 0.5 as nessecary
    ax.set_ylim(0,0.5) ### Uncomment when we're using a real array, not A, for plotting
    ax.hist(data[row,:])

hist = ax.hist(rankmatrix[0,:])

anim = animation.FuncAnimation(fig, update_hist, rankmatrix.shape[0], fargs=(rankmatrix, ) )

Writer = animation.writers['ffmpeg']
writer = Writer(fps=1)
anim.save('hist.mp4', writer=writer)
