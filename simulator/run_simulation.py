from quantum_simulator import QuantumSimulator

if __name__ == "__main__":
    simulator = QuantumSimulator(num_qubits=2)
    
    # Apply Hadamard to qubit 0
    simulator.apply_gate("H", [0])

    # Apply CNOT gate (control: qubit 0, target: qubit 1)
    simulator.apply_gate("CNOT", [0, 1])

    # Run simulation and measure
    result = simulator.run()
    
    print(f"Measurement result: {result}")
