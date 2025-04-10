import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import os

class QuantumSimulator:
    def __init__(self, num_qubits):
        self.n = num_qubits
        self.state = np.zeros((2**num_qubits, 1), dtype=complex)
        self.state[0, 0] = 1

    def apply_hadamard(self, qubit):
        H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
        self.state = self.apply_single_qubit_gate(H, qubit)

    def apply_single_qubit_gate(self, gate, qubit):
        dim = 2 ** self.n
        full_gate = np.eye(dim, dtype=complex)
        for i in range(dim):
            if (i >> qubit) & 1 == 0:
                full_gate[i, i] = gate[0, 0]
                full_gate[i, i ^ (1 << qubit)] = gate[0, 1]
                full_gate[i ^ (1 << qubit), i] = gate[1, 0]
                full_gate[i ^ (1 << qubit), i ^ (1 << qubit)] = gate[1, 1]
        return full_gate @ self.state

    def measure(self):
        probabilities = np.abs(self.state) ** 2
        probabilities = probabilities.ravel()
        result = np.random.choice(len(probabilities), p=probabilities)
        return bin(result)[2:].zfill(self.n)

class QuantumFourierTransform:
    def __init__(self, num_qubits, sim):
        self.num_qubits = num_qubits
        self.sim = sim

    def apply_qft(self):
        for i in range(self.num_qubits):
            self.sim.apply_hadamard(i)
            for j in range(i + 1, self.num_qubits):
                theta = np.pi / (2 ** (j - i))
                self.apply_controlled_phase(i, j, theta)
        self.swap_registers()

    def apply_controlled_phase(self, control, target, theta):
        new_state = self.sim.state.copy()
        for i in range(len(new_state)):
            if ((i >> control) & 1 == 1) and ((i >> target) & 1 == 1):
                new_state[i] *= np.exp(1j * theta)
        self.sim.state = new_state

    def swap_registers(self):
        for i in range(self.num_qubits // 2):
            self.swap(i, self.num_qubits - i - 1)

    def swap(self, q1, q2):
        new_state = self.sim.state.copy()
        for i in range(len(new_state)):
            j = i ^ ((1 << q1) | (1 << q2))
            if i < j:
                new_state[i], new_state[j] = new_state[j], new_state[i]
        self.sim.state = new_state

    def visualize_top_states(self, top_k=5, threshold=0.01):
        probabilities = np.abs(self.sim.state) ** 2
        states = [f"|{i:0{self.num_qubits}b}>" for i in range(len(probabilities))]

        filtered = [(s, p[0]) for s, p in zip(states, probabilities) if p > threshold]
        filtered.sort(key=lambda x: x[1], reverse=True)
        top_states = filtered[:top_k]

        if not top_states:
            top_states = sorted(zip(states, probabilities.ravel()), key=lambda x: x[1], reverse=True)[:top_k]

        labels, probs = zip(*top_states)
        plt.bar(labels, probs, color='skyblue', edgecolor='black')
        plt.title("Top QFT Output States")
        plt.xlabel("Quantum States")
        plt.ylabel("Probability")
        plt.tight_layout()
        plt.show()

# GUI for QFT
def run_qft_gui():
    def execute_qft():
        try:
            n = int(qubit_entry.get())
            if n < 1 or n > 10:
                raise ValueError("Number of qubits must be between 1 and 10")

            sim = QuantumSimulator(n)
            qft = QuantumFourierTransform(n, sim)
            qft.apply_qft()
            result = sim.measure()

            output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "qft_output.txt"))
            with open(output_path, "w") as f:
                f.write(result)

            result_var.set(f"Measured state: |{result}>")

            if visualize_var.get() == 1:
                qft.visualize_top_states()

            messagebox.showinfo("Success", f"Result saved to {output_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    root = tk.Tk()
    root.title("Quantum Fourier Transform")
    root.geometry("400x350")
    root.configure(bg="#1e1e2e")

    tk.Label(root, text="Quantum Fourier Transform", font=("Helvetica", 16), bg="#1e1e2e", fg="white").pack(pady=10)
    tk.Label(root, text="Enter number of qubits: (e.g., : 3)", bg="#1e1e2e", fg="white").pack()
    qubit_entry = tk.Entry(root, justify="center", bg="#2a2b3a", fg="white", insertbackground="white")
    qubit_entry.insert(0, "")
    qubit_entry.pack(pady=10)

    visualize_var = tk.IntVar()
    tk.Checkbutton(root, text="Show Top Output Probabilities", variable=visualize_var, bg="#1e1e2e", fg="white",
                   activebackground="#1e1e2e", activeforeground="white", selectcolor="#1e1e2e").pack(pady=5)

    tk.Button(root, text="Run QFT", command=execute_qft, bg="#0a84ff", fg="white").pack(pady=5)
    tk.Button(root, text="Close", command=root.destroy, bg="#444654", fg="white").pack(pady=5)

    result_var = tk.StringVar()
    tk.Label(root, textvariable=result_var, bg="#1e1e2e", fg="lightgreen", wraplength=350).pack(pady=15)

    root.mainloop()


if __name__ == "__main__":
    run_qft_gui()