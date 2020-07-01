# %% codecell
# initialize
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.extensions import UnitaryGate
from qiskit.visualization import plot_histogram
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import scipy.linalg as la

# %% codecell
# define gates

def coin(n_qubits):
    had = np.array([[1,1],[1,-1]])/np.sqrt(2)
    return np.kron(had, np.identity(2**(n_qubits-1)))

def step(n_qubits):

    add_mat = np.zeros([2**(n_qubits-1),2**(n_qubits-1)])
    add_mat[0,-1] = 1
    for i in range(2**(n_qubits-1)-1):
        add_mat[i+1,i]=1

    sub_mat = np.zeros([2**(n_qubits-1),2**(n_qubits-1)])
    sub_mat[-1,0] = 1
    for i in range(2**(n_qubits-1)-1):
        sub_mat[i,i+1]=1

    up_proj = np.diag([1,0])
    down_proj = np.diag([0,1])

    step = np.kron(down_proj,add_mat) + np.kron(up_proj,sub_mat)

    return step

# %% codecell
# run circuit

n_qubits = 5 # INCLUDES COIN
n_steps = 3 # number of timesteps

C = coin(n_qubits)
S = step(n_qubits)
SC = S @ C

psi_init = np.zeros(2**n_qubits)
# start in the middle of the line
psi_init[2**(n_qubits-2)] = 1

psi_list = [SC @ psi_init]

for i in range(n_steps-1):
    psi_list.append(SC @ psi_list[-1])

# fig, ax = plt.subplots(1,2)
# ax[0].matshow(np.real(np.diag(psi_list[-1])))
# ax[1].matshow(np.imag(np.diag(psi_list[-1])))
# psi_list[-1][:2**(n_qubits-1)]
# psi_list[-1][2**(n_qubits-1):]

# %% codecell
# find probabilities after each step
probabilityList = []
for psi in psi_list:
    probabilities = [np.abs(psi[i])**2 + np.abs(psi[i + 2**(n_qubits-1)])**2
     for i in range(2**(n_qubits-1))]
    probabilityList.append(probabilities)

# plot probabilities at the final step
plt.plot(np.arange(2**(n_qubits-1)) - 2**(n_qubits-2), probabilityList[-1])
