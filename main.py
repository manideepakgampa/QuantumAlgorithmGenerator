
import importlib
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai_model.models.nlp_model import NLPModel
from ai_model.models.neural_network import NeuralNetwork

# Load dataset
dataset_path = os.path.join("data", "dataset.csv")
dataset = pd.read_csv(dataset_path)

# Initialize models
nlp_model = NLPModel(dataset)
nn_model = NeuralNetwork()

def execute_algorithm(algorithm_name, extracted_features):
    """Dynamically imports and runs the selected quantum algorithm."""
    try:
        # Import the selected algorithm module
        module = importlib.import_module(f"quantum_algorithms.{algorithm_name.lower()}")
        
        # Run the algorithm's main function and return the result
        if hasattr(module, "run"):
            result = module.run(extracted_features)
            return result
        else:
            return "❌ Error: Algorithm module does not have a 'run' function."
    except ImportError:
        return f"❌ Error: Algorithm '{algorithm_name}' not implemented."

def main():
    print("🔍 Enter your problem description:")
    user_query = input("> ")

    # Step 1: Classify problem type using NLP model
    problem_type, extracted_features = nlp_model.classify(user_query)

    # Step 2: Predict algorithm using Neural Network
    predicted_algorithm = nn_model.predict(problem_type, extracted_features)

    print(f"✅ Selected Quantum Algorithm: {predicted_algorithm}")

    # Step 3: Execute the selected quantum algorithm
    output = execute_algorithm(predicted_algorithm, extracted_features)
    
    print(f"🔹 Solution Output: {output}")

if __name__ == "__main__":
    main()
