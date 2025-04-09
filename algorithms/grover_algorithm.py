import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator.quantum_simulator import QuantumSimulator

class GroverAlgorithm:
    def __init__(self, num_qubits, target_state):
        self.num_qubits = num_qubits
        self.num_states = 2 ** num_qubits
        self.target_state = target_state
        self.state_vector = np.full((self.num_states,), 1 / np.sqrt(self.num_states))

    def apply_oracle(self):
        self.state_vector[self.target_state] *= -1

    def apply_diffusion(self):
        mean = np.mean(self.state_vector)
        self.state_vector = 2 * mean - self.state_vector

    def run(self, iterations):
        for _ in range(iterations):
            self.apply_oracle()
            self.apply_diffusion()
        return np.argmax(self.state_vector ** 2)

    def visualize(self):
        probabilities = np.abs(self.state_vector) ** 2
        states = [f"|{i:0{self.num_qubits}b}>" for i in range(self.num_states)]
        plt.bar(states, probabilities, color='blue')
        plt.xlabel("Quantum States")
        plt.ylabel("Probability")
        plt.title("Grover’s Algorithm Result")
        plt.show()

# -------------------------------
# GUI Section
# -------------------------------

def launch_gui():
    def run_grover():
        try:
            qubits = int(qubit_entry.get())
            target = int(target_entry.get())
            iterations = int(iter_entry.get())

            if target >= 2 ** qubits:
                raise ValueError("Target state exceeds number of representable states.")

            grover = GroverAlgorithm(qubits, target)
            result = grover.run(iterations)
            result_var.set(f"Measured state: |{result:0{qubits}b}>")

            # Path to save the result
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            output_path = os.path.join(project_root, "grover_output.txt")

            # Save result
            with open(output_path, "w") as f:
                f.write(f"{result}")
            print(f"✅ Result saved to: {output_path}")


            print(f"✅ Result saved to {output_path}")

            if visualize_var.get() == 1:
                grover.visualize()

            messagebox.showinfo("Saved", f"Result saved: |{result:0{qubits}b}>")
        except Exception as e:
            messagebox.showerror("Runtime Error", str(e))
            sys.exit(0)

    root = tk.Tk()
    root.title("Grover's Algorithm")
    root.geometry("420x460")
    root.configure(bg="#1e1e2e")

    tk.Label(root, text="Grover's Algorithm", font=("Helvetica", 16, "bold"), fg="white", bg="#1e1e2e").pack(pady=10)

    tk.Label(root, text="Number of Qubits:", bg="#1e1e2e", fg="white").pack()
    qubit_entry = tk.Entry(root, fg="white", bg="#2a2b3a", justify="center", insertbackground="white")
    qubit_entry.insert(0, "e.g., 3")
    qubit_entry.pack(pady=5)

    tk.Label(root, text="Target State (as integer):", bg="#1e1e2e", fg="white").pack()
    target_entry = tk.Entry(root, fg="white", bg="#2a2b3a", justify="center", insertbackground="white")
    target_entry.insert(0, "e.g., 5")
    target_entry.pack(pady=5)

    tk.Label(root, text="Number of Iterations:", bg="#1e1e2e", fg="white").pack()
    iter_entry = tk.Entry(root, fg="white", bg="#2a2b3a", justify="center", insertbackground="white")
    iter_entry.insert(0, "e.g., 2")
    iter_entry.pack(pady=5)

    visualize_var = tk.IntVar()
    tk.Checkbutton(
        root, text="Visualize Final States", variable=visualize_var,
        onvalue=1, offvalue=0, selectcolor="#1e1e2e",
        bg="#1e1e2e", fg="white", activebackground="#1e1e2e", activeforeground="white"
    ).pack(pady=5)

    result_var = tk.StringVar()
    tk.Label(root, textvariable=result_var, wraplength=350, bg="#1e1e2e", fg="lightgreen").pack(pady=10)

    tk.Button(root, text="Run", command=run_grover, bg="#0a84ff", fg="white").pack(pady=5)
    tk.Button(root, text="Close", command=root.destroy, bg="#444654", fg="white").pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    try:
        launch_gui()
        sys.exit(0)
    except Exception as e:
        print("Error launching GUI:", e)
        sys.exit(0)
