# %% codecell
# initialize
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.extensions import UnitaryGate
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# %% codecell
# make a unitary gate

# random hermitian matrix
ham = np.matrix(np.random.random((4,4)))
ham = ham + ham.H
# make its unitary
unit_mat = np.matrix(la.expm(1j*ham))
# unitary gate object
unit_gate = UnitaryGate(unit_mat, label='test')

########################
# I don't think this block works, but it may be useful ClassicalRegister
########################

# # %% codecell
# # make a circuit with this gate
#
# testckt=unit_gate.definition
# testckt.name = 'test'
# testckt.draw(output='text')

# %% codecell
# setup circuit

q = QuantumRegister(4)
c = ClassicalRegister(4)
qc = QuantumCircuit(q,c)

qc.h(1)
qc.x(2)
# add on subcircuit from unitary
qc.append(unit_gate, [0,1])
qc.measure([0,1,2,3], [0,1,2,3])

# draw circuit
qc.draw(output="mpl")

# %% codecell
# decompose ckt and draw

qc.decompose().draw(output='mpl')
