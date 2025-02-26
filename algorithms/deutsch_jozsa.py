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
        - If the function is balanced, it applies a phase flip to half the states.
        """
        if self.function_type == "balanced":
            # Flip the phase of half the states
            for i in range(2 ** (self.n - 1)):  # Flip only half of them
                self.sim.apply_phase_flip(format(int(i), f'0{self.n}b'))

    def run(self):
        """
        Implements the Deutsch-Jozsa algorithm.
        """
        print(f"🚀 Running Deutsch-Jozsa Algorithm with {self.n} qubits ({self.function_type} function)")
        
        # Step 1: Apply Hadamard to all qubits
        self.sim.apply_hadamard(range(self.n))
        
        # Step 2: Apply Oracle
        self.oracle()
        
        # Step 3: Apply Hadamard to all qubits again
        self.sim.apply_hadamard(range(self.n))

        # Step 4: Measure the result
        result = self.sim.measure()
        
        # If all qubits are |0>, the function is constant. Otherwise, it's balanced.
        if result == "0" * self.n:
            print("✅ Function is CONSTANT")
        else:
            print("✅ Function is BALANCED")
class DeutschJozsaSimulator(QuantumSimulator):
    def measure(self):
        """Custom measurement: Extract first n-1 bits (Function is constant or balanced)"""
        measured_state = super().measure()
        extracted_bits = measured_state[:self.num_qubits-1]  # Ignore last qubit
        print(f"🧐 Custom Measurement for Deutsch-Jozsa: {extracted_bits}")
        return extracted_bits

# Example Usage
dj = DeutschJozsa(n=3, function_type="balanced")  # Change to "constant" for testing
dj.run()
