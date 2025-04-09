import math
import random
import sys
import os
import tkinter as tk
from tkinter import messagebox

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator.quantum_simulator import QuantumSimulator

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def modular_exponentiation(base, exponent, mod):
    result = 1
    base = base % mod
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % mod
        exponent = exponent >> 1
        base = (base * base) % mod
    return result

def quantum_order_finding(a, N):
    sim = QuantumSimulator(3)
    for qubit in range(3):
        sim.apply_hadamard(qubit)
    measured_state = sim.measure()
    r = int(measured_state, 2)
    if r <= 1 or r % 2 != 0:
        r += 2
    return r

def factorize(N):
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

        for _ in range(5):
            a = random.randint(2, N - 1)
            g = gcd(a, N)
            if g > 1 and g < N:
                factors.append(g)
                recursive_factorization(N // g)
                return

            r = quantum_order_finding(a, N)
            if r % 2 != 0:
                continue

            x = modular_exponentiation(a, r // 2, N)
            if x == -1 or x % N == 0:
                continue

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

        if N > 1 and N != original_N:
            factors.append(N)

    recursive_factorization(N)
    return sorted(factors)

def launch_gui():
    def run_algorithm():
        try:
            number = int(entry.get())
            if number <= 1:
                raise ValueError("Enter a number > 1")
            result = factorize(number)
            result_text.set(f"Factors of {number}: {', '.join(map(str, result))}")
            # Save to file for Flask
            with open("shor_output.txt", "w") as f:
                f.write(f"{number}:{','.join(map(str, result))}")
        except ValueError as ve:
            messagebox.showerror("Invalid Input", str(ve))

    root = tk.Tk()
    root.title("Shor's Algorithm")
    root.geometry("400x300")
    root.configure(bg="#1e1e2e")

    tk.Label(root, text="Shor's Algorithm", font=("Helvetica", 16, "bold"), fg="white", bg="#1e1e2e").pack(pady=10)
    tk.Label(root, text="Enter a number (N):", font=("Helvetica", 12), bg="#1e1e2e", fg="white").pack()

    global entry
    entry = tk.Entry(root, font=("Helvetica", 12), justify="center", bg="#2a2b3a", fg="white")
    entry.pack(pady=10)

    global result_text
    result_text = tk.StringVar()
    tk.Label(root, textvariable=result_text, wraplength=350, font=("Helvetica", 11), bg="#1e1e2e", fg="lightgreen").pack(pady=10)

    tk.Button(root, text="Run", command=run_algorithm, font=("Helvetica", 12), bg="#0a84ff", fg="white").pack(pady=5)
    tk.Button(root, text="Close", command=root.destroy, font=("Helvetica", 10), bg="#444654", fg="white").pack()

    root.mainloop()

if __name__ == "__main__":
    launch_gui()
