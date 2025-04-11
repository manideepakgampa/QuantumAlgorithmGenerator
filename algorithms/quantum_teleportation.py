import numpy as np
import tkinter as tk
from tkinter import messagebox, scrolledtext
import os
import sys

# Fix for local path loading (use absolute project structure in real file)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator.quantum_simulator import QuantumSimulator

class QuantumTeleportation:
    def __init__(self, alpha=None, beta=None):
        self.sim = QuantumSimulator(3)
        self.log = ""

        # Normalize user input or use random values
        if alpha is not None and beta is not None:
            norm = np.sqrt(alpha ** 2 + beta ** 2)
            self.alpha, self.beta = alpha / norm, beta / norm
        else:
            a, b = np.random.rand(2)
            norm = np.sqrt(a ** 2 + b ** 2)
            self.alpha, self.beta = a / norm, b / norm

    def create_entanglement(self):
        self.log += "🔗 Creating Entangled Pair...\n"
        self.sim.apply_hadamard(1)
        self.sim.apply_cnot(1, 2)

    def teleport(self):
        self.log = "🚀 Starting Quantum Teleportation Protocol...\n"
        self.log += f"🎯 Prepared State |ψ⟩ = {self.alpha:.3f}|0⟩ + {self.beta:.3f}|1⟩\n"

        # Properly initialize |ψ⟩ on 3-qubit system
        self.sim.state = np.zeros((8, 1), dtype=complex)
        self.sim.state[0] = self.alpha  # |000⟩
        self.sim.state[1] = self.beta   # |001⟩

        self.create_entanglement()
        self.sim.apply_cnot(0, 1)
        self.sim.apply_hadamard(0)

        measurements = self.sim.measure()
        self.log += f"📏 Alice's Measurement: {measurements}\n"

        if measurements[-2] == "1":
            self.sim.apply_x(2)
            self.log += "💡 Bob applied X correction\n"
        if measurements[-1] == "1":
            self.sim.apply_z(2)
            self.log += "💡 Bob applied Z correction\n"

        result = self.sim.measure()
        final_bit = result[-1]
        self.log += f"\n✅ Teleported State at Bob's Qubit: |{final_bit}⟩\n🎉 Teleportation Complete!\n"

        # Save result
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "teleportation_output.txt"))
        with open(output_path, "w") as f:
            f.write(final_bit)

        return self.log, final_bit

# ---------------- GUI ---------------- #

def run_teleportation_gui():
    def execute_teleportation():
        try:
            alpha_val = alpha_entry.get().strip()
            beta_val = beta_entry.get().strip()

            alpha = float(alpha_val) if alpha_val else None
            beta = float(beta_val) if beta_val else None

            qt = QuantumTeleportation(alpha=alpha, beta=beta)
            log, result = qt.teleport()

            output_box.delete("1.0", tk.END)
            output_box.insert(tk.END, log)
            messagebox.showinfo("Success", f"Teleportation complete. Bob's qubit is: |{result}⟩")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    root = tk.Tk()
    root.title("Quantum Teleportation")
    root.geometry("620x550")
    root.configure(bg="#1e1e2e")

    tk.Label(root, text="Quantum Teleportation Protocol", font=("Helvetica", 16), bg="#1e1e2e", fg="white").pack(pady=10)

    form_frame = tk.Frame(root, bg="#1e1e2e")
    form_frame.pack(pady=5)

    tk.Label(form_frame, text="Amplitude α (|0⟩):", bg="#1e1e2e", fg="white").grid(row=0, column=0, padx=5, sticky="e")
    alpha_entry = tk.Entry(form_frame, bg="#2a2b3a", fg="white", insertbackground="white", justify="center")
    alpha_entry.grid(row=0, column=1, padx=5)

    tk.Label(form_frame, text="Amplitude β (|1⟩):", bg="#1e1e2e", fg="white").grid(row=1, column=0, padx=5, sticky="e")
    beta_entry = tk.Entry(form_frame, bg="#2a2b3a", fg="white", insertbackground="white", justify="center")
    beta_entry.grid(row=1, column=1, padx=5)

    tk.Button(root, text="Run Teleportation", command=execute_teleportation, bg="#0a84ff", fg="white").pack(pady=10)
    tk.Button(root, text="Close", command=root.destroy, bg="#444654", fg="white").pack(pady=5)

    output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=75, height=18, bg="#2a2b3a", fg="white", font=("Courier", 10))
    output_box.pack(padx=10, pady=10)

    root.mainloop()

# Run GUI if called directly
if __name__ == "__main__":
    run_teleportation_gui()
