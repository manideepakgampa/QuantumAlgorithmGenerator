#ai_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from quantum_algorithms import (
    shors_algorithm, grovers_algorithm, quantum_phase_estimation, hhl_algorithm, qaoa_algorithm,
    vqe_algorithm, deutsch_jozsa_algorithm, simons_algorithm, quantum_walk_algorithm, bernstein_vazirani_algorithm
)

# Define the Feedforward Neural Network
class QuantumAlgorithmNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QuantumAlgorithmNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Define available algorithms
quantum_algorithms = [
    shors_algorithm, grovers_algorithm, quantum_phase_estimation, hhl_algorithm, qaoa_algorithm,
    vqe_algorithm, deutsch_jozsa_algorithm, simons_algorithm, quantum_walk_algorithm, bernstein_vazirani_algorithm
]

algorithm_names = [
    "Shor’s Algorithm", "Grover’s Algorithm", "Quantum Phase Estimation", "HHL Algorithm", "QAOA Algorithm",
    "VQE Algorithm", "Deutsch-Jozsa Algorithm", "Simon’s Algorithm", "Quantum Walk Algorithm", "Bernstein-Vazirani Algorithm"
]

# Create AI Model
def create_model():
    input_size = 5  # Define based on query features
    hidden_size = 10
    output_size = len(quantum_algorithms)
    model = QuantumAlgorithmNN(input_size, hidden_size, output_size)
    return model

# Load trained model (or initialize new one)
model = create_model()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Process user query and predict the best quantum algorithm
def process_query(query_vector):
    model.eval()
    with torch.no_grad():
        query_tensor = torch.tensor([query_vector], dtype=torch.float32)
        output = model(query_tensor)
        predicted_index = torch.argmax(output).item()
        return algorithm_names[predicted_index], quantum_algorithms[predicted_index]
