import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator.quantum_simulator import QuantumSimulator
class GroverAlgorithm:
    def __init__(self, num_qubits, target_state):
        self.num_qubits = num_qubits
        self.num_states = 2**num_qubits
        self.target_state = target_state
        self.state_vector = np.full((self.num_states,), 1/np.sqrt(self.num_states))  # Uniform superposition

    def apply_oracle(self):
        """Applies the Oracle: Inverts the amplitude of the target state"""
        self.state_vector[self.target_state] *= -1  # Phase flip
        print(f"📌 Phase flip applied to state |{self.target_state:0{self.num_qubits}b}>")

    def apply_diffusion(self):
        """Applies the Diffusion Operator: Inverts amplitudes about the mean"""
        mean = np.mean(self.state_vector)
        self.state_vector = 2 * mean - self.state_vector
        print("🔄 Inversion about mean applied")

    def run(self, iterations):
        """Runs Grover’s algorithm for a given number of iterations"""
        print(f"\n🚀 Running Grover’s Algorithm for {iterations} Iterations on {self.num_qubits} qubits\n")
        
        for i in range(iterations):
            print(f"🔄 Iteration {i + 1}:")
            
            # Apply Oracle
            self.apply_oracle()
            
            # Apply Diffusion Operator
            self.apply_diffusion()

        # Measure the final state
        measured_state = np.argmax(self.state_vector**2)
        print(f"\n🎯 Measured state: |{measured_state:0{self.num_qubits}b}>\n")
        return measured_state

    def visualize(self):
        """Displays a histogram of the state probabilities"""
        probabilities = np.abs(self.state_vector) ** 2
        states = [f"|{i:0{self.num_qubits}b}>" for i in range(self.num_states)]

        plt.bar(states, probabilities, color='blue')
        plt.xlabel("Quantum States")
        plt.ylabel("Probability")
        plt.title("Final State Probabilities after Grover’s Algorithm")
        plt.show()
class GroverSimulator(QuantumSimulator):
    def measure(self):
        """Custom measurement: Extract the most probable state"""
        measured_state = super().measure()
        probable_state = measured_state  # Simulate finding the correct state
        print(f"🧐 Custom Measurement for Grover's: {probable_state}")
        return probable_state

# Example Usage
num_qubits = 3
target_state = 5  # Target state is |101>
iterations = 2  # Optimal number of iterations for 3 qubits

# grover = GroverAlgorithm(num_qubits, target_state)
# result = grover.run(iterations)
# grover.visualize()
