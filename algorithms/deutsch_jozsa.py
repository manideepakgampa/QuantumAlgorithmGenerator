import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator.quantum_simulator import QuantumSimulator  # Import your custom simulator

class DeutschJozsa:
    def __init__(self, n, function_type):
        self.n = n  # Number of qubits
        self.sim = QuantumSimulator(n)  # Initialize quantum simulator
        self.function_type = function_type  # 'constant' or 'balanced'
    
    def oracle(self):
        """
        Implements the oracle function:
        - If the function is constant, it does nothing.
        - If the function is balanced, it applies a phase flip (Z gate) to some qubits.
        """
        if self.function_type == "balanced":
            balanced_qubits = np.random.choice(range(self.n), size=self.n//2, replace=False)
            for qubit in balanced_qubits:
                print(f"⚡ Applying Phase Flip on qubit {qubit}")
                self.sim.apply_phase_flip(qubit)

    def run(self):
        """
        Implements the Deutsch-Jozsa algorithm.
        """
        print(f"🚀 Running Deutsch-Jozsa Algorithm with {self.n} qubits ({self.function_type} function)")
        
        # Step 1: Apply Hadamard to all qubits
        for qubit in range(self.n):
            self.sim.apply_hadamard(qubit)

        # Step 2: Apply Oracle
        self.oracle()
        
        # Step 3: Apply Hadamard to all qubits again
        for qubit in range(self.n):
            self.sim.apply_hadamard(qubit)

        # Step 4: Measure the result
        result = self.sim.measure()
        
        # If all qubits are |0>, the function is constant. Otherwise, it's balanced.
        if result == "0" * self.n:
            print("✅ Function is CONSTANT")
        else:
            print("✅ Function is BALANCED")

class DeutschJozsaSimulator(QuantumSimulator):
    def measure(self):
        probabilities = np.abs(self.state) ** 2  # Compute probabilities
        probabilities = probabilities.flatten()  # Ensure it's 1D\
        result = np.random.choice(len(probabilities), p=probabilities)
        return bin(result)[2:].zfill(self.n)  # Convert to binary string


# Example Usage
dj = DeutschJozsa(n=3, function_type="balanced")  # Change to "constant" for testing
dj.run()
