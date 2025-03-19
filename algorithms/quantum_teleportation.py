import numpy as np
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator.quantum_simulator import QuantumSimulator

class QuantumTeleportation:
    def __init__(self):
        self.sim = QuantumSimulator(3)  # Three qubits: |ψ⟩ (Alice), entangled pair (A, B)

    def create_entanglement(self):
        """Creates an entangled Bell pair between qubit 1 (Alice) and qubit 2 (Bob)."""
        print("\n🔗 Creating Entangled Pair...")
        self.sim.apply_hadamard(1)  # Put qubit 1 in superposition
        self.sim.apply_cnot(1, 2)  # Create entanglement between qubit 1 and qubit 2

    def teleport(self):
        """Performs the teleportation protocol."""
        print("\n🚀 Starting Quantum Teleportation...")

        # Step 1: Prepare a random state |ψ⟩ (Alice's qubit)
        alpha, beta = np.random.rand(2)
        norm = np.sqrt(alpha**2 + beta**2)
        alpha, beta = alpha / norm, beta / norm  # Normalize
        print(f"🎯 Random State |ψ⟩ = {alpha:.3f}|0⟩ + {beta:.3f}|1⟩ (Alice's Qubit)")

        # Encode this state into qubit 0
        self.sim.state_vector[0] = alpha
        self.sim.state_vector[1] = beta

        # Step 2: Create entanglement
        self.create_entanglement()

        # Step 3: Alice applies a CNOT gate and Hadamard gate
        self.sim.apply_cnot(0, 1)
        self.sim.apply_hadamard(0)

        # Step 4: Measure Alice's qubits (Qubit 0 and 1)
        alice_measurements = self.sim.measure()
        print(f"📏 Alice's Measurement Results: {alice_measurements}")

        # Step 5: Bob applies corrections based on Alice’s results
        if alice_measurements[-2] == "1":  
            self.sim.apply_pauli_x(2)  # Apply X if Alice's first bit is 1
            print("💡 Bob applied X correction")
        if alice_measurements[-1] == "1":
            self.sim.apply_pauli_z(2)  # Apply Z if Alice's second bit is 1
            print("💡 Bob applied Z correction")

        # Step 6: Final measurement on Bob's qubit
        bob_result = self.sim.measure()
        print(f"\n✅ Teleported State: {bob_result[-1]}")
        print("🎉 Quantum Teleportation Successful!")

# Run the teleportation protocol
if __name__ == "__main__":
    teleportation = QuantumTeleportation()
    teleportation.teleport()
