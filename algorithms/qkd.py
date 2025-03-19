import numpy as np
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator.quantum_simulator import QuantumSimulator

class QKD_BB84:
    def __init__(self, num_qubits=8):
        self.num_qubits = num_qubits
        self.sim = QuantumSimulator(num_qubits)  # Use your quantum simulator
        self.alice_bits = []
        self.alice_bases = []
        self.bob_bases = []
        self.bob_results = []

    def generate_alice_bits(self):
        """Alice generates random bits and random bases."""
        self.alice_bits = [random.choice([0, 1]) for _ in range(self.num_qubits)]
        self.alice_bases = [random.choice(['Z', 'X']) for _ in range(self.num_qubits)]
        print(f"🎲 Alice's Bits:     {self.alice_bits}")
        print(f"🎭 Alice's Bases:    {self.alice_bases}")

    def encode_qubits(self):
        """Alice encodes bits in chosen bases using the quantum simulator."""
        for i in range(self.num_qubits):
            if self.alice_bits[i] == 1:
                self.sim.apply_phase_flip(i)  # Flip |0⟩ → |1⟩
            if self.alice_bases[i] == 'X':  
                self.sim.apply_hadamard([i])  # Change basis to Hadamard (X basis)

    def measure_bob_qubits(self):
        """Bob randomly selects bases and measures qubits."""
        self.bob_bases = [random.choice(['Z', 'X']) for _ in range(self.num_qubits)]
        print(f"🎭 Bob's Bases:      {self.bob_bases}")

        for i in range(self.num_qubits):
            if self.bob_bases[i] == 'X':  
                self.sim.apply_hadamard([i])  # Convert back to standard basis before measuring

        self.bob_results = list(map(int, self.sim.measure()))  # Convert measurement to integer list
        print(f"📏 Bob's Results:    {self.bob_results}")

    def sift_key(self):
        """Alice and Bob compare bases and keep bits where bases match."""
        sifted_key = []
        for i in range(self.num_qubits):
            if self.alice_bases[i] == self.bob_bases[i]:  # Keep only matching bases
                sifted_key.append(self.bob_results[i])

        print(f"🔑 Final Shared Key: {sifted_key}")
        return sifted_key

    def run(self):
        """Runs the full BB84 QKD protocol."""
        print("\n🔐 Running BB84 Quantum Key Distribution...\n")
        self.generate_alice_bits()
        self.encode_qubits()
        self.measure_bob_qubits()
        shared_key = self.sift_key()
        print("\n✅ QKD Protocol Complete!\n")
        return shared_key


# Run QKD Protocol
if __name__ == "__main__":
    qkd = QKD_BB84(num_qubits=8)  # Adjust qubit count as needed
    qkd.run()
