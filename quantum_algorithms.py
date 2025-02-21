#quantum_algorithms.py
from qiskit import QuantumCircuit, Aer, execute
from qiskit_algorithms import Shor, Grover, HHL, VQE, QAOA
from qiskit.circuit.library import QFT
from qiskit.quantum_info import SparsePauliOp  # Replaces PauliSumOp
import numpy as np

# ------------------- 1. Shor’s Algorithm -------------------
def shors_algorithm(N):
    """Performs integer factorization using Shor’s Algorithm."""
    shor = Shor(Aer.get_backend('qasm_simulator'))
    result = shor.factor(N)
    return result

# ------------------- 2. Grover’s Algorithm -------------------
def grovers_algorithm(n):
    """Searches an unstructured database using Grover’s Algorithm."""
    qc = QuantumCircuit(n)
    grover = Grover(qc)
    result = execute(grover, Aer.get_backend('qasm_simulator')).result()
    return result.get_counts()

# ------------------- 3. Quantum Phase Estimation (QPE) -------------------
def quantum_phase_estimation(unitary, precision=4):
    """Estimates the phase of a unitary operator."""
    qc = QuantumCircuit(precision + 1, precision)
    qc.h(range(precision))  # Apply Hadamard gates
    qc.append(unitary, [precision] + list(range(precision)))  # Apply unitary
    qc.append(QFT(precision).inverse(), range(precision))  # Apply inverse QFT
    qc.measure(range(precision), range(precision))
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator).result()
    return result.get_counts()

# ------------------- 4. HHL Algorithm -------------------
def hhl_algorithm(A, b):
    """Solves Ax = b using the HHL quantum algorithm."""

    from qiskit.utils import algorithm_globals
    algorithm_globals.random_seed = 42
    hhl = HHL()
    solution = hhl.solve(A, b)
    return solution

# ------------------- 5. Quantum Approximate Optimization Algorithm (QAOA) -------------------
def qaoa_algorithm(problem_instance):
    """Solves combinatorial optimization problems using QAOA."""
    hamiltonian = SparsePauliOp.from_list(problem_instance)
    qaoa = QAOA()
    result = qaoa.compute_minimum_eigenvalue(hamiltonian)
    return result



# ------------------- 6. Variational Quantum Eigensolver (VQE) -------------------
def vqe_algorithm(hamiltonian):
    """Finds the lowest eigenvalue of a Hamiltonian using VQE."""
    
    from qiskit.circuit.library import TwoLocal
    ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
    vqe = VQE(ansatz)
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    return result

# ------------------- 7. Deutsch-Jozsa Algorithm -------------------
def deutsch_jozsa_algorithm(oracle, n):
    """Determines whether a function is constant or balanced."""
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)  
    qc.h(range(n + 1))
    qc.append(oracle, range(n + 1))
    qc.h(range(n))
    qc.measure(range(n), range(n))
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator).result()
    return result.get_counts()

# ------------------- 8. Simon’s Algorithm -------------------
def simons_algorithm(oracle, n):
    """Finds the hidden period of a function."""
    qc = QuantumCircuit(2 * n, n)
    qc.h(range(n))
    qc.append(oracle, range(2 * n))
    qc.h(range(n))
    qc.measure(range(n), range(n))
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator).result()
    return result.get_counts()

# ------------------- 9. Quantum Walk Algorithm -------------------
def quantum_walk_algorithm(n):
    """Simulates a quantum walk on a graph."""
    qc = QuantumCircuit(n)
    qc.h(0)
    for _ in range(n):
        qc.h(range(n))
        qc.cx(0, 1)
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator).result()
    return result.get_counts()

# ------------------- 10. Bernstein-Vazirani Algorithm -------------------
def bernstein_vazirani_algorithm(secret_string):
    """Finds a hidden binary string in a function."""
    n = len(secret_string)
    qc = QuantumCircuit(n + 1, n)
    qc.h(range(n))
    qc.x(n)
    qc.h(n)
    for i in range(n):
        if secret_string[i] == "1":
            qc.cx(i, n)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator).result()
    return result.get_counts()

# ------------------- Test the Functions -------------------
if __name__ == "__main__":
    print("Shor’s Algorithm Factorization of 15:", shors_algorithm(15))
    print("Grover’s Algorithm Search Results:", grovers_algorithm(4))
