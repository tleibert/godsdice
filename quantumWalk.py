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
def qft(n_qubits):

    qc = QuantumCircuit(n_qubits, name='QFT')

    for i in range(n_qubits):
        qc.h(-1-i)
        for j in range(1,n_qubits-i):
            qc.cu1(pi/(2**j), -1-(i+j), -1-i)
        qc.barrier([i for i in range(n_qubits)])

    # usually you would swap the order of the qubits here - since we'll be
    # inverting the QFT at the end, it's cheaper to leave that out and just swap
    # the manipulations in between QFT and QFT_dag

    return qc

def step(n_qubits):
    # if heads (|0>) move +1, if tails (|1>), -1
    # n_qubits INCLUDES the coin qubit - coin is the last one

    qc = QuantumCircuit(n_qubits, name='Step')
    theta = 2*pi/(2**(n_qubits-1))

    for i in range(n_qubits-1):
        qc.rz(-(2**i)*theta, n_qubits-i-2)
        qc.crz(2*(2**i)*theta, n_qubits-1, n_qubits-i-2)
    # fudge phase factor
    # qc.rz(theta, -1)

    return qc

# step written out with matrix multiplication instead of QFT
def adder_step(n_qubits):

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

    return UnitaryGate(step)

# %% codecell
# Quantum Walk circuit

n_qubits = 8 # THIS INCLUDES THE COIN - coin is the last qubit
n_steps = 50 # number of timesteps

# pick either 'statevector_simulator' or 'qasm_simulator'
backend = 'statevector_simulator'

qc = QuantumCircuit(n_qubits, n_qubits-1)
# start in the middle of the chain, coin heads up
qc.x(-2)
qc.u2(3*pi/2,pi/2,-1)

qc.append(qft(n_qubits-1), [i for i in range(n_qubits-1)])
for i in range(n_steps):
    # flip the coin
    qc.h(-1)
    # take a step
    qc.append(step(n_qubits), [i for i in range(n_qubits)])
qc.append(qft(n_qubits-1).inverse(), [i for i in range(n_qubits-1)])
if backend is 'qasm_simulator':
    qc.measure([i for i in range(n_qubits-1)], [i for i in range(n_qubits-1)])

simulator = Aer.get_backend(backend)
job = execute(qc, simulator)

if backend is 'qasm_simulator':
    counts_ex = job.result().get_counts()
    # make the histogram include zeros
    counts = {}
    for i in range(2**(n_qubits-1)):
        key = bin(i)[2:].zfill(n_qubits-1)
        counts[key] = 0
    for key in counts_ex.keys():
        counts[key] = counts_ex[key]
    plot_histogram(counts, bar_labels=False)

if backend is 'statevector_simulator':
    psi_final = job.result().get_statevector()
    # combine probability of x & heads with x & tails
    prob_final = [np.abs(psi_final[i])**2 + np.abs(psi_final[i + 2**(n_qubits-1)])**2
     for i in range(2**(n_qubits-1))]
    plt.plot(np.arange(2**(n_qubits-1)) - 2**(n_qubits-2), prob_final)

    plt.matshow(np.real(np.diag(psi_final)))
    plt.matshow(np.imag(np.diag(psi_final)))
