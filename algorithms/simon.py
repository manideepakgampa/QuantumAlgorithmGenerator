import sys
import os
import random
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.quantum_simulator import QuantumSimulator

class SimonsAlgorithm:
    def __init__(self, n, hidden_string):
        self.n = n  # Number of qubits
        self.hidden_string = hidden_string  # Secret period s
        self.sim = QuantumSimulator(n * 2)  # 2n qubits (input + output)

    def oracle(self, x):
        """Simulated Oracle ensuring f(x) = f(x ⊕ s)"""
        y = int(x, 2) ^ int(self.hidden_string, 2)
        return format(y, f'0{self.n}b')  # Ensure binary format

    def run(self):
        print(f"🚀 Running Simon's Algorithm with {self.n} qubits (Hidden string: {self.hidden_string})")

        collected_equations = []
        max_attempts = 50  # Avoid infinite loops

        for _ in range(max_attempts):
            self.sim.apply_hadamard(range(self.n))  # Create superposition
            x = self.sim.measure()[:self.n]  # Measure first n qubits
            y = self.oracle(x)  # Apply oracle transformation
            equation = format(int(x, 2) ^ int(y, 2), f'0{self.n}b')

            # Ensure the equation is new and linearly independent
            if equation != "0" * self.n and self.is_independent(collected_equations, equation):
                collected_equations.append(equation)
                print(f"🧐 Debug: Measured Equation {equation}")

            # Stop if we collected enough independent equations
            if len(collected_equations) >= self.n - 1:
                break

        # If we still don't have enough, print a warning
        if len(collected_equations) < self.n - 1:
            print(f"⚠️ Warning: Only collected {len(collected_equations)} equations instead of {self.n - 1}")

        # Solve for s
        s = self.solve_equations(collected_equations)
        print(f"✅ Hidden String Found: {s}")

    def is_independent(self, equations, new_eq):
        """Check if new_eq is linearly independent from existing equations (mod 2)."""
        if not equations:
            return True  # Always accept the first equation

        matrix = np.array([[int(bit) for bit in eq] for eq in equations + [new_eq]], dtype=int)
        rank_before = np.linalg.matrix_rank(matrix[:-1])  # Rank before adding
        rank_after = np.linalg.matrix_rank(matrix)  # Rank after adding

        return rank_after > rank_before  # If rank increases, it's independent

    def solve_equations(self, equations):
        """Solve system of linear equations mod 2 using Gaussian elimination."""
        matrix = np.array([[int(bit) for bit in eq] for eq in equations], dtype=int)
        n = len(matrix[0])  # Number of variables

        # Gaussian elimination mod 2
        for i in range(n):
            for j in range(i, len(matrix)):
                if matrix[j][i] == 1:
                    matrix[[i, j]] = matrix[[j, i]]  # Swap rows
                    break

            for j in range(i + 1, len(matrix)):
                if matrix[j][i] == 1:
                    matrix[j] = (matrix[j] + matrix[i]) % 2  # XOR rows mod 2

        # Extract the last row as solution
        s = "".join(str(matrix[-1][i]) for i in range(n))
        return s


# Example Usage
if __name__ == "__main__":
    simon = SimonsAlgorithm(n=4, hidden_string="1011")  # Example with 4 qubits
    simon.run()
