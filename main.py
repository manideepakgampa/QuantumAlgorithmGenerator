from quantum_algorithms import shor_algorithm, grover_algorithm, qaoa_algorithm, hhl_algorithm, vqe_algorithm

def process_query(query_vector):
    """Maps query vector to a quantum algorithm."""
    algorithms = {
        (1, 0, 0, 0, 0): ("Shor's Algorithm", shor_algorithm),
        (0, 1, 0, 0, 0): ("Grover's Algorithm", grover_algorithm),
        (0, 0, 1, 0, 0): ("QAOA (Quantum Approximate Optimization Algorithm)", qaoa_algorithm),
        (0, 0, 0, 1, 0): ("HHL (Solving Linear Equations)", hhl_algorithm),
        (0, 0, 0, 0, 1): ("VQE (Eigenvalue Estimation)", vqe_algorithm)
    }
    
    query_tuple = tuple(query_vector)  # Convert list to tuple for dictionary lookup
    return algorithms.get(query_tuple, ("Unknown Algorithm", lambda x: "No valid algorithm found."))

def main():
    print("Enter your problem type (example: factorization, search, optimization, etc.):")
    problem_type = input().strip().lower()
    
    # Simple encoding for problem type
    problem_encoding = {
        "factorization": [1, 0, 0, 0, 0],
        "search": [0, 1, 0, 0, 0],
        "optimization": [0, 0, 1, 0, 0],
        "linear equations": [0, 0, 0, 1, 0],
        "eigenvalue estimation": [0, 0, 0, 0, 1]
    }
    
    query_vector = problem_encoding.get(problem_type, [0, 0, 0, 0, 0])
    
    algorithm_name, algorithm_function = process_query(query_vector)
    print(f"Recommended Algorithm: {algorithm_name}")
    
    # Run the recommended algorithm
    print("Executing...")
    result = algorithm_function(4)  # Example input value
    print("Result:", result)

if __name__ == "__main__":
    main()
