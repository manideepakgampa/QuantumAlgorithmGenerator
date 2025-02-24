from quantum_circuit import QuantumCircuit

class QuantumSimulator:
    def __init__(self, num_qubits):
        self.circuit = QuantumCircuit(num_qubits)

    def apply_gate(self, gate_name, qubits):
        self.circuit.add_gate(gate_name, qubits)

    def run(self):
        self.circuit.execute()
        return self.circuit.measure()
