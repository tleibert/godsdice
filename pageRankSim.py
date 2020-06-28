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

# %% codecell

# adjacency matrix
A = np.array([  [0,1,0,0,1,1,0,0],
                [0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,1,0],
                [0,0,1,0,1,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,1,0,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0]])

# make C and S operators
C = Testing_Unitaries.constructC(A)
# C = np.matrix(C)

c_gate = UnitaryGate(C)

S = Testing_Unitaries.constructS(A)
# S = np.matrix(S)

s_gate = UnitaryGate(S)

########################
# I don't think this block works, but it may be useful later
########################

# # %% codecell
# # make a circuit with this gate
#
# testckt=unit_gate.definition
# testckt.name = 'test'
# testckt.draw(output='text')
########################

# %% codecell
# run circuit

# I'm running this for an absurdly long time

# initialize to superposition without |up>|111> or |down>|111>
init_vector = np.ones(16)/np.sqrt(14)
init_vector[7] = init_vector[15] = 0

# run the circuits for many timesteps and average the results
runs = [10*i + 50 for i in range(46)]
counts = {'000':0,'001':0,'010':0,'011':0,'100':0,'101':0,'110':0,'111':0}
for run in runs:

    q = QuantumRegister(4)
    c = ClassicalRegister(3)
    qc = QuantumCircuit(q,c)

    qc.initialize(init_vector, [0,1,2,3])

    for i in range(run):
        qc.append(c_gate, [0,1,2,3])
        qc.append(s_gate, [0,1,2,3])
    # add on subcircuit from unitary
    qc.measure([0,1,2], [0,1,2])

    simulator = Aer.get_backend("qasm_simulator")
    job = execute(qc, simulator, shots=5000)
    result = job.result()
    counts_run = result.get_counts(qc)
    for key in counts_run.keys():
        counts[key] += counts_run[key]
plot_histogram(counts)

# draw circuit
# qc.draw(output="mpl")

############################
# DECOMPOSING UNITARY NOT WORKING
############################
#
# decomp = QuantumCircuit(4,3)
# decomp.append(c_gate.definition, [0,1,2,3])
# decomp.append(s_gate.definition, [0,1,2,3])
# decomp.decompose().draw(output='mpl')
# %% codecell
# decompose ckt and draw
