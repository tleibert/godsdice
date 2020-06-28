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
######################
# MAKE SURE Testing_Unitaries IS UP TO DATE!
######################

# %% codecell
# create the operators

# input adjacency matrix
A = np.array([  [0,1,0,0,1,1,0,0],
                [0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,1,0],
                [0,0,1,0,1,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,1,0,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0]])

# number of qubits needed for graph
n_qubits = int(np.log2(A.shape[0]))

# make C and S operators
C = Testing_Unitaries.constructC(A)
c_gate = UnitaryGate(C)

S = Testing_Unitaries.constructS(A)
s_gate = UnitaryGate(S)

# %% codecell
# run circuit

# initialize to superposition without |up>|111> or |down>|111>
init_vector = np.ones(16)/np.sqrt(14)
init_vector[7] = init_vector[15] = 0

# run the circuits for many timesteps and average the results
# the paper started at 50 and ran up to 200-500, depending on the size of the
# graph
# I'm doing it in steps of 10
runs = [10*i + 50 for i in range(46)]
# all the probabilities are added up here to yield the quantum rank
rank = {'000':0,'001':0,'010':0,'011':0,'100':0,'101':0,'110':0}
for run in runs:

    # initialize qc
    qc = QuantumCircuit(n_qubits+1,n_qubits)

    qc.initialize(init_vector, [i for i in range(n_qubits+1)])

    # add on appropriate #'s of C and S gates
    for i in range(run):
        qc.append(c_gate, [i for i in range(n_qubits+1)])
        qc.append(s_gate, [i for i in range(n_qubits+1)])

    # measure output probabilities
    # we're 90% sure that the last qubit is the coin
    qc.measure([i for i in range(n_qubits)], [i for i in range(n_qubits)])

    # run circuit and store output probabilities
    simulator = Aer.get_backend("qasm_simulator")
    job = execute(qc, simulator, shots=5000)
    result = job.result()
    counts_run = result.get_counts(qc)
    # add up probabilities for all runs
    for key in counts_run.keys():
        rank[key] += counts_run[key]

# normalize & plot quantum rank
plot_histogram(rank)

# %% codecell
############################
# DECOMPOSING UNITARY NOT WORKING
############################
#
# decomp = QuantumCircuit(4,3)
# decomp.append(c_gate.definition, [0,1,2,3])
# decomp.append(s_gate.definition, [0,1,2,3])
# decomp.decompose().draw(output='mpl')

# # %% codecell
########################
# I don't think this block works, but it may be useful later
# Makes a subcircuit out of a UnitaryGate
########################

# # make a circuit with this gate
#
# testckt=unit_gate.definition
# testckt.name = 'test'
# testckt.draw(output='text')
########################
