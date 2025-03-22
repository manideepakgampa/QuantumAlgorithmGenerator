import math
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator.quantum_simulator import QuantumSimulator  # Import Quantum Simulator

def gcd(a, b):
    """Compute the Greatest Common Divisor (GCD)."""
    while b:
        a, b = b, a % b
    return a

def modular_exponentiation(base, exponent, mod):
    """Compute (base^exponent) % mod using fast modular exponentiation."""
    result = 1
    base = base % mod
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % mod
        exponent = exponent >> 1
        base = (base * base) % mod
    return result

def quantum_order_finding(a, N):
    """Simulate the quantum order finding using a Quantum Simulator."""
    sim = QuantumSimulator(3)  # Initialize a quantum simulator with 3 qubits

    for qubit in range(3):  
        sim.apply_hadamard(qubit)  # Apply Hadamard to all qubits
    
    measured_state = sim.measure()  # Simulate measurement
    r = int(measured_state, 2)  # Convert binary measurement to integer

    if r <= 1 or r % 2 != 0:  # Ensure even order r
        r += 2
    return r

def factorize(N):
    """Fully factorize N using quantum-assisted order finding."""
    factors = []
    original_N = N 
    def recursive_factorization(N):
        if N <= 1:
            return
        if N % 2 == 0:  
            while N % 2 == 0:
                factors.append(2)
                N //= 2
            recursive_factorization(N)
            return

        for _ in range(5):  # Retry multiple times for robustness
            a = random.randint(2, N - 1)
            g = gcd(a, N)
            if g > 1 and g < N:  # Ensure g is a proper factor
                factors.append(g)
                recursive_factorization(N // g)  # Factorize the quotient
                return

            r = quantum_order_finding(a, N)  # Use quantum simulator
            if r % 2 != 0:
                continue  # Retry if r is odd

            x = modular_exponentiation(a, r // 2, N)
            if x == -1 or x % N == 0:
                continue  # Avoid trivial solutions

            factor1 = gcd(x - 1, N)
            factor2 = gcd(x + 1, N)

            if 1 < factor1 < N:
                factors.append(factor1)
                recursive_factorization(N // factor1)
                return

            if 1 < factor2 < N:
                factors.append(factor2)
                recursive_factorization(N // factor2)
                return

        # 🛠 FIX: Avoid adding N itself if it hasn't been broken down
        if N > 1 and N != original_N:  # Check to prevent returning N itself
            factors.append(N)  # Add remaining prime factor


    recursive_factorization(N)
    return sorted(factors)  # Return sorted factors

if __name__ == "__main__":
    N = 123456  # Example number
    result = factorize(N)
    if result:
        print("Shor's Algorithm for Integer Factorization")
        print(f"Factors of {N} are: {result}")
    else:
        print("Factorization failed. Try again.")
