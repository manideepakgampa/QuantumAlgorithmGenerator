import numpy as np
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator.quantum_simulator import QuantumSimulator

class BB84:
    def __init__(self, num_qubits=10):
        self.num_qubits = num_qubits
        self.sim = QuantumSimulator(num_qubits)

    def generate_random_bits(self):
        """Generate a random bit sequence for key generation."""
        return [random.choice([0, 1]) for _ in range(self.num_qubits)]

    def generate_random_bases(self):
        """Generate random basis choices: 0 for standard (Z), 1 for Hadamard (X)."""
        return [random.choice([0, 1]) for _ in range(self.num_qubits)]

    def encode_qubits(self, bits, bases):
        """Encodes bits into qubits using chosen bases."""
        for i in range(self.num_qubits):
            if bases[i] == 1:  # If Hadamard basis is chosen
                self.sim.apply_hadamard([i])
            if bits[i] == 1:  # Flip qubit if bit is 1
                self.sim.apply_phase_flip(i)

    def measure_qubits(self, bases):
        """Measures qubits in Bob's chosen bases."""
        for i in range(self.num_qubits):
            if bases[i] == 1:  # Measure in Hadamard basis
                self.sim.apply_hadamard([i])
        return self.sim.measure()

    def sift_keys(self, alice_bits, alice_bases, bob_bases, bob_measurements):
        """Key sifting process where mismatched bases are discarded."""
        shared_key = []
        for i in range(self.num_qubits):
            if alice_bases[i] == bob_bases[i]:  # Only keep bits where bases match
                shared_key.append(alice_bits[i])
        return shared_key

    def run(self):
        """Executes the BB84 protocol."""
        print("\n🔑 Running BB84 Quantum Key Distribution...\n")

        # Step 1: Alice prepares random bits and bases
        alice_bits = self.generate_random_bits()
        alice_bases = self.generate_random_bases()
        print(f"Alice's Bits:   {alice_bits}")
        print(f"Alice's Bases:  {alice_bases}")

        # Step 2: Alice encodes and sends qubits
        self.encode_qubits(alice_bits, alice_bases)

        # Step 3: Bob randomly selects bases
        bob_bases = self.generate_random_bases()
        print(f"Bob's Bases:    {bob_bases}")

        # Step 4: Bob measures qubits
        bob_measurements = self.measure_qubits(bob_bases)
        print(f"Bob's Bits:     {bob_measurements}")

        # Step 5: Key sifting (discard mismatched bases)
        key = self.sift_keys(alice_bits, alice_bases, bob_bases, bob_measurements)
        print(f"🔐 Shared Secret Key: {key}")

# Execute BB84
if __name__ == "__main__":
    qkd = BB84(num_qubits=10)
    qkd.run()
