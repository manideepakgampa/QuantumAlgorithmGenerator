#Quantum Fourier Transform (QFT)

import numpy as np

class QuantumFourierTransform:
    def __init__(self, num_qubits, sim):
        self.num_qubits = num_qubits
        self.sim = sim  # Use the QuantumSimulator

    def apply_qft(self):
        """Applies QFT to all qubits."""
        for i in range(self.num_qubits):
            self.sim.apply_hadamard([i])  # Apply Hadamard gate
            for j in range(i + 1, self.num_qubits):
                theta = np.pi / (2 ** (j - i))
                self.sim.apply_controlled_phase(i, j, theta)  # Apply controlled phase shift
        self.swap_registers()

    def swap_registers(self):
        """Reverses the order of qubits (QFT requires bit-reversal)."""
        for i in range(self.num_qubits // 2):
            self.sim.swap(i, self.num_qubits - i - 1)

# Example usage
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from simulator.quantum_simulator import QuantumSimulator
    sim = QuantumSimulator(4)  # Example with 4 qubits
    qft = QuantumFourierTransform(4, sim)
    qft.apply_qft()
    sim.measure()  # Measure final state
