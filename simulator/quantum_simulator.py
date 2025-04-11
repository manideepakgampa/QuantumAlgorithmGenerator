import numpy as np

class QuantumSimulator:
    def __init__(self, num_qubits):
        """Initialize the quantum state |0...0> for given number of qubits."""
        self.n = num_qubits
        self.state = np.zeros((2**num_qubits,1), dtype=complex)
        self.state[0,0] = 1  # Start in |0...0> state

    def apply_hadamard(self, qubit):
        """Applies the Hadamard gate to a single qubit."""
        H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
        self.state = self.apply_single_qubit_gate(H, qubit)

    def apply_x(self, qubit):
        """Applies the Pauli-X (NOT) gate to a single qubit."""
        X = np.array([[0, 1], [1, 0]])
        self.state = self.apply_single_qubit_gate(X, qubit)

    def apply_z(self, qubit):
        """Applies the Pauli-Z gate to a single qubit."""
        Z = np.array([[1, 0], [0, -1]])
        self.state = self.apply_single_qubit_gate(Z, qubit)

    def apply_cnot(self, control, target):
        """Applies the CNOT gate with given control and target qubits."""
        new_state = self.state.copy()
        for i in range(len(self.state)):
            # If control qubit is 1, swap the target qubit
            if (i >> control) & 1 == 1:
                target_index = i ^ (1 << target)  # Flip target qubit
                new_state[i], new_state[target_index] = new_state[target_index], new_state[i]
        self.state = new_state

    def apply_single_qubit_gate(self, gate, qubit):
        """Applies a single-qubit gate to the given qubit index."""
        assert 0 <= qubit < self.n, f"Invalid qubit index: {qubit}"

        dim = 2 ** self.n
        full_gate = np.eye(dim, dtype=complex)  # Start with identity matrix

        # Apply gate
        for i in range(dim):
            if (i >> qubit) & 1 == 0:
                full_gate[i, i] = gate[0, 0]
                full_gate[i, i ^ (1 << qubit)] = gate[0, 1]
                full_gate[i ^ (1 << qubit), i] = gate[1, 0]
                full_gate[i ^ (1 << qubit), i ^ (1 << qubit)] = gate[1, 1]

        # Debugging print before multiplication
        # print(f"🛠 DEBUG: full_gate shape: {full_gate.shape}")
        # print(f"🛠 DEBUG: self.state shape: {self.state.shape}")
        # print(f"🛠 DEBUG: Applying gate on qubit {qubit}")

        updated_state = full_gate @ self.state  # Apply the gate

        if updated_state is None:
            raise ValueError("❌ Error: Matrix multiplication resulted in None!")

        return updated_state 


    def measure(self):
        probabilities = np.abs(self.state) ** 2  # Compute probabilities
        probabilities = probabilities.ravel()  # Convert to 1D array

        # Normalize to avoid floating-point sum issues
        probabilities /= probabilities.sum()

        result = np.random.choice(len(probabilities), p=probabilities)
        return bin(result)[2:].zfill(self.n)  # Convert to binary string


    def apply_phase_flip(self, qubit):
        """Applies a phase flip (Z gate) to the given qubit."""
        print(f"⚡ Applying Phase Flip on qubit {qubit}")

        Z = np.array([[1, 0], [0, -1]])  # Phase Flip (Pauli-Z Gate)
        self.state = self.apply_single_qubit_gate(Z, qubit)  # Apply gate


# ✅ Test the simulator with CNOT
if __name__ == "__main__":
    sim = QuantumSimulator(2)
    sim.apply_hadamard(0)  # Put qubit 0 in superposition
    sim.apply_cnot(0, 1)   # Apply CNOT with control = 0, target = 1
    print("Measured State:", sim.measure())
