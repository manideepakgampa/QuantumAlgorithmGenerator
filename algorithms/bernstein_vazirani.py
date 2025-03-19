import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator.quantum_simulator import QuantumSimulator

class BernsteinVazirani:
    def __init__(self, n, hidden_string):
        self.n = n  # Number of qubits
        self.hidden_string = hidden_string  # Secret bit string
        self.sim = QuantumSimulator(n)  # Extra qubit for oracle

    def oracle(self):
        """Encodes the hidden string into the quantum state"""
        for i, bit in enumerate(self.hidden_string):
            if bit == '1':
                self.sim.apply_phase_flip(format(i, f'0{self.n}b'))  



    def run(self):
        """Runs the Bernstein-Vazirani algorithm"""
        
        # Apply Hadamard to all input qubits
        for qubit in range(self.n):
            self.sim.apply_hadamard(qubit)
        
        self.oracle()  # Apply the oracle (Hidden String encoding)
        
        # ✅ No need to check for state initialization!
        
        # Apply Hadamard again after the oracle
        for qubit in range(self.n):
            self.sim.apply_hadamard(qubit)


        measured_state = self.sim.measure()
        print(f"🧐 Debug: Full Measured State: {measured_state}")
        print(f"✅ Hidden String Found: {measured_state[-self.n:]}")  # Correct slicing


# Example run
if __name__ == "__main__":
    print("🔄 Script is executing...")  # Add this
    print("Bernstein-Vazirani Algorithm for n=4 and hidden string '1011'")
    bv = BernsteinVazirani(4, "1011")
    bv.run()

