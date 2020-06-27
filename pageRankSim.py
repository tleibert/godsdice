from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

q = QuantumRegister(4)
c = ClassicalRegister(4)
qc = QuantumCircuit(q,c)

qc.draw('mpl')
