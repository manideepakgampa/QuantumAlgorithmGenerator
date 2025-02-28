import math
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.quantum_simulator import QuantumSimulator 


def gcd(a, b):
    """Calculate the Greatest Common Divisor (GCD)."""
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
    """Simulated quantum order finding using our QuantumSimulator."""
    sim = QuantumSimulator(2)  # Initialize the simulator with 2 qubits

    full_measurement = sim.measure()  # Get full measured state
    r = full_measurement[:2]  # Extract first 2 qubits manually

    sim.apply_hadamard([0, 1])  # Superposition
    full_measurement = sim.measure()  # Measure again after Hadamard
    measured_r = full_measurement[:2]  # Extract first 2 qubits manually

    # Simulate order finding (since real order finding needs full quantum simulation)
    r = random.randint(2, N - 1)
    if r % 2 != 0:  # Ensure it's even
        r += 1

    return r


def shor_algorithm(N):
    """Shor's Algorithm for integer factorization."""
    if N % 2 == 0:
        return 2, N // 2  # Simple case for even numbers

    # Step 1: Choose a random number `a` coprime to `N`
    a = random.randint(2, N - 1)
    g = gcd(a, N)
    
    if g > 1:
        return g, N // g  # Found a factor

    # Step 2: Quantum Order Finding (Simulated)
    r = quantum_order_finding(a, N)
    if r % 2 != 0:
        return None  # Failed, restart

    # Step 3: Compute possible factors
    factor1 = gcd(modular_exponentiation(a, r // 2, N) - 1, N)
    factor2 = gcd(modular_exponentiation(a, r // 2, N) + 1, N)

    if factor1 > 1 and factor1 < N:
        return factor1, N // factor1
    if factor2 > 1 and factor2 < N:
        return factor2, N // factor2

    return None  # Factorization failed

class ShorQuantumSimulator(QuantumSimulator):
    def measure(self, qubits):
        """Custom measurement for Shor's Algorithm"""
        measured_state = super().measure()  # Get default measurement
        selected_bits = "".join(measured_state[q] for q in qubits)
        print(f"🧐 Custom Measurement for Shor: {selected_bits}")
        return selected_bits

# Example Usage
if __name__ == "__main__":
    N = 15 # Example number
    result = shor_algorithm(N)
    if result:
        print(f"Factors of {N} are: {result}")
    else:
        print("Factorization failed. Try again.")
