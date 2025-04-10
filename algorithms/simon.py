import numpy as np
import tkinter as tk
from tkinter import messagebox
import os
import sys
import random

# Ensure simulator is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator.quantum_simulator import QuantumSimulator

class SimonsAlgorithm:
    def __init__(self, n, hidden_string):
        self.n = n
        self.hidden_string = hidden_string
        self.sim = QuantumSimulator(n * 2)

    def oracle(self, x):
        y = int(x, 2) ^ int(self.hidden_string, 2)
        return format(y, f'0{self.n}b')

    def run(self):
        collected_equations = []
        max_attempts = 50

        for _ in range(max_attempts):
            for i in range(self.n):
                self.sim.apply_hadamard(i)
            x = self.sim.measure()[:self.n]
            y = self.oracle(x)
            equation = format(int(x, 2) ^ int(y, 2), f'0{self.n}b')

            if equation != "0" * self.n and self.is_independent(collected_equations, equation):
                collected_equations.append(equation)

            if len(collected_equations) >= self.n - 1:
                break

        if len(collected_equations) < self.n - 1:
            print(f"⚠️ Only collected {len(collected_equations)} equations instead of {self.n - 1}")

        s = self.solve_equations(collected_equations)
        return s

    def is_independent(self, equations, new_eq):
        if not equations:
            return True
        matrix = np.array([[int(bit) for bit in eq] for eq in equations + [new_eq]], dtype=int)
        rank_before = np.linalg.matrix_rank(matrix[:-1])
        rank_after = np.linalg.matrix_rank(matrix)
        return rank_after > rank_before

    def solve_equations(self, equations):
        matrix = np.array([[int(bit) for bit in eq] for eq in equations], dtype=int)
        n = len(matrix[0])
        for i in range(n):
            for j in range(i, len(matrix)):
                if matrix[j][i] == 1:
                    matrix[[i, j]] = matrix[[j, i]]
                    break
            for j in range(i + 1, len(matrix)):
                if matrix[j][i] == 1:
                    matrix[j] = (matrix[j] + matrix[i]) % 2
        s = "".join(str(matrix[-1][i]) for i in range(n))
        return s

# Tkinter GUI
def run_simons_gui():
    def execute_simons():
        try:
            n = int(qubit_entry.get())
            hidden = hidden_entry.get().strip()
            if len(hidden) != n or not all(c in '01' for c in hidden):
                raise ValueError("Hidden string must be a binary string of length equal to number of qubits")

            simon = SimonsAlgorithm(n=n, hidden_string=hidden)
            result = simon.run()

            # Save output
            output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simons_output.txt"))
            with open(output_path, "w") as f:
                f.write(result)

            result_var.set(f"Hidden string found: {result}")
            messagebox.showinfo("Success", f"Simon's algorithm completed. Output saved.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    root = tk.Tk()
    root.title("Simon's Algorithm")
    root.geometry("400x350")
    root.configure(bg="#1e1e2e")

    tk.Label(root, text="Simon's Algorithm", font=("Helvetica", 16), bg="#1e1e2e", fg="white").pack(pady=10)

    tk.Label(root, text="Number of Qubits (n): (e.g. : 4)", bg="#1e1e2e", fg="white").pack()
    qubit_entry = tk.Entry(root, justify="center", bg="#2a2b3a", fg="white", insertbackground="white")
    qubit_entry.insert(0, "")
    qubit_entry.pack(pady=10)

    tk.Label(root, text="Hidden String (binary): (e.g., : 1011)", bg="#1e1e2e", fg="white").pack()
    hidden_entry = tk.Entry(root, justify="center", bg="#2a2b3a", fg="white", insertbackground="white")
    hidden_entry.insert(0, "")
    hidden_entry.pack(pady=10)

    tk.Button(root, text="Run Algorithm", command=execute_simons, bg="#0a84ff", fg="white").pack(pady=10)
    tk.Button(root, text="Close", command=root.destroy, bg="#444654", fg="white").pack(pady=5)

    result_var = tk.StringVar()
    tk.Label(root, textvariable=result_var, bg="#1e1e2e", fg="lightgreen", wraplength=350).pack(pady=15)

    root.mainloop()

if __name__ == "__main__":
    try:
        run_simons_gui()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)