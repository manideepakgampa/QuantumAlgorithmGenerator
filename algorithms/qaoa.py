import numpy as np 
import sys
import networkx as nx
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator.quantum_simulator import QuantumSimulator


class QAOA:
    def __init__(self, graph, p=1):
        """
        QAOA implementation for Max-Cut problem.
        :param graph: NetworkX Graph object
        :param p: Number of QAOA layers
        """
        self.graph = graph  # Store the actual graph object
        self.n = graph.number_of_nodes()  # Get number of nodes
        self.p = p  # Depth of QAOA
        self.sim = QuantumSimulator(self.n)  # Initialize simulator
        self.edges = list(graph.edges)  # Extract edges from Graph object

    def apply_cost_hamiltonian(self, gamma):
        """Applies the cost Hamiltonian using controlled phase gates."""
        for i, j in self.edges:
            print(f"⚡ Applying CPhase on qubits {i}-{j} with γ = {gamma:.4f}")
            self.sim.apply_cphase(i, j, gamma)

    def apply_mixing_hamiltonian(self, beta):
        """Applies the mixing Hamiltonian using X rotations."""
        for i in range(self.n):
            self.sim.apply_rx(i, 2 * beta)

    def initialize_state(self):
        """Initializes the state into an equal superposition."""
        if self.sim.state is None or len(self.sim.state) == 0:
            self.sim.initialize_state()  # Ensure the state is initialized
        for i in range(self.n):
            self.sim.apply_hadamard(i)  # ✅ Pass as a list

    def run(self, gamma, beta):
        """Executes QAOA circuit with given parameters."""
        self.initialize_state()

        # Apply p layers of QAOA
        for layer in range(self.p):
            print(f"🌀 QAOA Layer {layer+1}/{self.p}")
            self.apply_cost_hamiltonian(gamma)
            self.apply_mixing_hamiltonian(beta)

        # Measure and return results
        return self.sim.measure()


# ✅ Correcting the graph definition
graph = nx.Graph()
graph.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Create a networkx graph

qaoa = QAOA(graph, p=1)

# Example Parameters (to be optimized)
gamma, beta = np.pi / 4, np.pi / 8
result = qaoa.run(gamma, beta)

print("📏 Measured Bitstring:", result)
