# Quantum Walk and Page Rank

This repo contains code to simulate a quantum random walk, the Quantum Page Rank algorithm, and visualize the output of those simulations.

classical_walk gives the results of R repeats of a classical random walk of N steps.

quantumWalk simulates a quantum walk using qasm_simulator, and returns the counts dictionary from the simulation result. It also includes the option to introduce random measurements during the quantum walk, which causes decoherence.

unitary_methods generates the coin and step operators for the Quantum Page Rank algorithm, given a network's adjacency matrix. It also includes code to generate adjacency matrices for binary trees and scale free networks.

pageRankSim contains a couple relevant methods:
->beginsim takes in an adjacency matrix and number of frames, and calls the simulate method, returning a matrix of frequency values where each row of the matrix contains data for a histogram, with n_frames rows, and n_states columns, where n_states is the total number of quantum states spanned by the graph's state space that is being animated. hist_visualization and network_visualization call this method to gather data to be animated.
->simulate is the method called by beginsim to simulate the circuits on qasm_simulator. it takes in an adjacency matrix, initializaton vector, number of runs, shots, and number of states. It returns a vector of rank information

hist_visualization takes in an adjacency matrix, calls pageRankSim, and creates an animation, showing the page rank of each node changing over time. network_visualization also takes in an adjacency matrix, and makes an animation showing the ranks displayed on the network itself. Nodes with a higher rank are colored lighter, and those with a lower rank are colored darker. Both visualization files contain an example at the bottom showing how to use them.

numpyQuantumWalk and numpyPageRankSim perform those simulations with matrix multiplication in numpy, rather than with qiskit's qasm_simulator.
