import importlib
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ai_model')))

from ai_model.models.nlp_model import NLPModel
from ai_model.models.neural_network import FeedforwardNN


# Load dataset for training NLP model
dataset_path = os.path.join("data", "dataset.csv")
dataset = pd.read_csv(dataset_path)

# Initialize the NLP and Neural Network models
nlp_model = NLPModel(dataset)  # Assuming this model uses the dataset for training
nn_model = FeedforwardNN()  # Neural Network should be pre-trained or trainable

def execute_algorithm(algorithm_name, extracted_features):
    """Dynamically imports and runs the selected quantum algorithm."""
    try:
        # Import the selected algorithm module dynamically
        module = importlib.import_module(f"quantum_algorithms.{algorithm_name.lower()}")
        
        # Run the algorithm's main function and return the result
        if hasattr(module, "run"):
            result = module.run(extracted_features)  # Assuming features are passed to the algorithm
            return result
        else:
            return "❌ Error: Algorithm module does not have a 'run' function."
    except ImportError:
        return f"❌ Error: Algorithm '{algorithm_name}' not implemented."

def main():
    print("🔍 Enter your problem description:")
    user_query = input("> ")

    # Step 1: Classify problem type and extract features using NLP model
    problem_type, extracted_features = nlp_model.classify(user_query)

    # Output the problem type and extracted features (for debugging)
    print(f"✅ Problem Type: {problem_type}")
    print(f"✅ Extracted Features: {extracted_features}")

    # Step 2: Predict the quantum algorithm using the Neural Network model
    predicted_algorithm = nn_model.predict(problem_type, extracted_features)

    print(f"✅ Selected Quantum Algorithm: {predicted_algorithm}")

    # Step 3: Execute the selected quantum algorithm
    output = execute_algorithm(predicted_algorithm, extracted_features)
    
    # Step 4: Display the result from the quantum algorithm
    print(f"🔹 Solution Output: {output}")

if __name__ == "__main__":
    main()
