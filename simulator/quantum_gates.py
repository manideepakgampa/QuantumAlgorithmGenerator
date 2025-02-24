import numpy as np

# Pauli Matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)  # NOT gate
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Hadamard Gate
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

# Phase Gates
S = np.array([[1, 0], [0, 1j]], dtype=complex)   # S Gate (π/2 phase)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # T Gate (π/4 phase)

# Controlled-NOT (CNOT) Gate
CNOT = np.array([[1, 0, 0, 0], 
                 [0, 1, 0, 0], 
                 [0, 0, 0, 1], 
                 [0, 0, 1, 0]], dtype=complex)

# Controlled-Z (CZ) Gate
CZ = np.array([[1, 0, 0, 0], 
               [0, 1, 0, 0], 
               [0, 0, 1, 0], 
               [0, 0, 0, -1]], dtype=complex)

# Gate dictionary
GATES = {
    "X": X,
    "Y": Y,
    "Z": Z,
    "H": H,
    "S": S,
    "T": T,
    "CNOT": CNOT,
    "CZ": CZ
}
