import numpy as np

class QuantumCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(2**num_qubits, dtype=complex)
        self.state_vector[0] = 1  # Initialize in |0...0> state
        self.gates = []  # Store gates for execution

    def add_gate(self, gate_name, qubits):
        """Adds a gate (Hadamard, X, etc.) to the circuit"""
        self.gates.append((gate_name, qubits))

    def add_custom_phase_flip(self, target_state):
        """
        Applies a phase flip to a given target state |x> → -|x|.
        :param target_state: The quantum state whose phase should be flipped.
        """
        target_index = int(target_state, 2)  # Convert binary state to integer index
        self.state_vector[target_index] *= -1  # Flip the phase
        print(f"Phase flip applied to |{target_state}>")

    def apply_inversion_about_mean(self):
        """Applies the inversion about the mean step in Grover’s diffusion operator"""
        mean_amplitude = np.mean(self.state_vector)
        self.state_vector = 2 * mean_amplitude - self.state_vector  # Reflect across the mean
        print("Inversion about mean applied")

    def execute(self):
        """Simulates execution by applying gates sequentially"""
        print("Executing quantum circuit with applied gates:")
        for gate, qubits in self.gates:
            print(f"Applying {gate} on qubits {qubits}")
            # TODO: Implement gate operations like Hadamard, X, CNOT, etc.

    def measure(self):
        """Simulates measurement of the quantum state"""
        probabilities = np.abs(self.state_vector) ** 2
        measured_state = np.random.choice(len(probabilities), p=probabilities)
        measured_state_bin = format(measured_state, f"0{self.num_qubits}b")
        print(f"Measured state: |{measured_state_bin}>")
        return measured_state_bin
