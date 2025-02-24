from qiskit import QuantumCircuit, Aer, transpile
from qiskit_aer import AerSimulator
from qiskit.utils import QuantumInstance
import numpy as np
from fractions import Fraction
import math

def qft(n):
    """Creates a Quantum Fourier Transform circuit on n qubits."""
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
        for j in range(i+1, n):
            qc.cp(np.pi / 2**(j-i), j, i)  # Controlled phase shift
    qc.barrier()
    for i in range(n//2):
        qc.swap(i, n-i-1)  # Reverse qubit order
    return qc

# Initialize Quantum Instance
quantum_instance = QuantumInstance(AerSimulator())

# ------------------- 1. Shor’s Algorithm -------------------
def shors_algorithm(N):
    """Shor’s Algorithm for integer factorization of N."""
    
    if N % 2 == 0:
        return f"Factors of {N}: 2 and {N // 2}"
    
    def gcd(a, b):
        """Finds the greatest common divisor."""
        while b:
            a, b = b, a % b
        return a
    
    def quantum_order_finding(N, a):
        """Uses Quantum Phase Estimation to find the order r of a (mod N)."""
        n = int(np.ceil(np.log2(N)))  # Number of qubits
        qc = QuantumCircuit(2 * n, n)
        
        # Initialize quantum registers
        qc.h(range(n))
        qc.x(n)
        
        # Apply modular exponentiation circuit
        for i in range(n):
            qc.append(modular_exponentiation(a, 2**i, N, 2*n), range(2*n))
        
        # Apply inverse QFT
        qc.append(qft(n).inverse(), range(n))
        
        # Measure
        qc.measure(range(n), range(n))
        
        # Run simulation
        simulator = AerSimulator()
        result = simulator.run(transpile(qc, simulator), shots=1).result()
        counts = result.get_counts()
        
        # Extract phase
        measured = max(counts, key=counts.get)
        phase = int(measured, 2) / 2**n
        fraction = Fraction(phase).limit_denominator(N)
        
        return fraction.denominator  # r

    def modular_exponentiation(a, exp, N, num_qubits):
        """Creates a circuit for modular exponentiation a^exp mod N."""
        qc = QuantumCircuit(num_qubits)
        for _ in range(exp):
            qc.unitary(np.array([[1, 0], [0, a % N]]), [0])  # Placeholder for mod operation
        return qc

    # Pick a random number a such that 1 < a < N
    a = np.random.randint(2, N)
    while gcd(a, N) != 1:
        a = np.random.randint(2, N)
    
    # Find the order of a mod N
    r = quantum_order_finding(N, a)
    
    if r % 2 == 1 or pow(a, r // 2, N) == N - 1:
        return "Failed to find factors, retrying..."
    
    # Compute factors
    factor1 = gcd(pow(a, r // 2, N) - 1, N)
    factor2 = gcd(pow(a, r // 2, N) + 1, N)

    if factor1 * factor2 == N:
        return f"Factors of {N}: {factor1} and {factor2}"
    else:
        return "Shor's Algorithm failed. Try again."

# Example Run
N = 15  # Example number to factor
print(shors_algorithm(N))
