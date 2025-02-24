import numpy as np

class QuantumRegister:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1  # Start in |0...0> state

    def apply_gate(self, gate, qubits):
        """
        Applies a quantum gate to the specified qubits.
        """
        num_states = 2**self.num_qubits
        new_state = np.zeros_like(self.state)

        for i in range(num_states):
            binary_i = format(i, f'0{self.num_qubits}b')  # Binary representation
            target_state = list(binary_i)

            # Apply single-qubit gate
            if len(qubits) == 1:
                qubit = qubits[0]
                zero_state = target_state.copy()
                one_state = target_state.copy()
                zero_state[qubit] = '0'
                one_state[qubit] = '1'
                zero_index = int("".join(zero_state), 2)
                one_index = int("".join(one_state), 2)
                new_state[i] = gate[0, 0] * self.state[zero_index] + gate[0, 1] * self.state[one_index]

        self.state = new_state

    def measure(self):
        probabilities = np.abs(self.state) ** 2
        return np.random.choice(len(self.state), p=probabilities)
