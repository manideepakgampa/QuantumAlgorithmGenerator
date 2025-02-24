from quantum_register import QuantumRegister
from quantum_gates import GATES

class QuantumCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.register = QuantumRegister(num_qubits)
        self.operations = []

    def add_gate(self, gate_name, qubits):
        if gate_name in GATES:
            self.operations.append((gate_name, qubits))
        else:
            raise ValueError(f"Gate {gate_name} not found!")

    def execute(self):
        for gate_name, qubits in self.operations:
            self.register.apply_gate(GATES[gate_name], qubits)

    def measure(self):
        return self.register.measure()
