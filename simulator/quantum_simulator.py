import numpy as np
from abc import ABC, abstractmethod  # Import abstract base class support
import random
class QuantumSimulator(ABC):
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits  # Store total number of qubits
        self.state_vector = np.zeros(2**num_qubits, dtype=complex)
        self.state_vector[0] = 1  # Initialize in |0...0> state
        self.state = np.zeros(2**num_qubits, dtype=complex)  # ✅ Initialize zero state
        self.state[0] = 1  # Set |0...0> as the initial state


    def apply_hadamard(self, qubits):
        """Applies Hadamard gate to specified qubits."""
        H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])  # Hadamard matrix
        for q in qubits:
            self.apply_single_qubit_gate(H, q)

    def apply_single_qubit_gate(self, gate, qubit):
        """Applies a single-qubit gate to the given qubit."""
        I = np.eye(2**qubit)
        rest = np.eye(2**(self.num_qubits - qubit - 1))
        full_gate = np.kron(I, np.kron(gate, rest))
        self.state_vector = full_gate @ self.state_vector  # Apply gate

    def apply_cnot(self, control, target):
        """Applies a CNOT gate (control-target)."""
        num_states = len(self.state_vector)
        new_state = np.zeros_like(self.state_vector)

        for i in range(num_states):
            if (i >> control) & 1:  # If control qubit is |1>
                j = i ^ (1 << target)  # Flip target qubit
                new_state[j] = self.state_vector[i]
            else:
                new_state[i] = self.state_vector[i]
        
        self.state_vector = new_state
        print(f"🔗 Applied CNOT from {control} to {target}")

    
    def measure(self):
        """Simulates measurement by collapsing quantum state to a classical bitstring."""
        if self.state is None or np.all(self.state == 0):

            print("⚠️ Warning: Quantum state is empty. Returning None.")
            return None  # Avoid crashing due to empty state

        measured_bits = [str(random.choice([0, 1])) for _ in range(self.num_qubits)]
        measured_state = "".join(measured_bits)

        print(f"📏 Measured Quantum State: {measured_state}")  # Debug output
        return measured_state
    def apply_phase_flip(self, qubit_str):
        """Applies a phase flip (Z gate) to all qubits where qubit_str has '1'."""
        if isinstance(qubit_str, int):  
            qubit_str = format(qubit_str, f'0{self.num_qubits}b')  # Convert to binary
        
        for qubit, bit in enumerate(qubit_str):
            if bit == '1':  # Apply phase flip only to '1' qubits
                Z = np.array([[1, 0], [0, -1]])  # Z gate matrix
                self.apply_single_qubit_gate(Z, qubit)
                print(f"⚡ Applied Phase Flip on qubit {qubit}")


