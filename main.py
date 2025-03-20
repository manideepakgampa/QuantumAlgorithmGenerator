import importlib
import pandas as pd
import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Move up to project root
from ai_model.models.nlp_model import NLPModel
from ai_model.models.neural_network import FeedforwardNN

# Define paths
dataset_path = r"C:\Users\manid\Projects\IQAD\data\dataset.csv"
model_path = r"C:\Users\manid\Projects\IQAD\ai_model\models\iqad_trained_model.pth"  # Trained PyTorch model

# Load dataset
dataset = pd.read_csv(dataset_path)

# Initialize the NLP Model
nlp_model = NLPModel(dataset_path)  

# Load trained PyTorch Neural Network Model
if os.path.exists(model_path):
    nn_model = FeedforwardNN(input_size=41, hidden_size=100, output_size=15) # Match output size with dataset
    nn_model.load_state_dict(torch.load(model_path))
    nn_model.eval()  # Set to evaluation mode
    print("✅ PyTorch Model loaded successfully!")
else:
    print("❌ Error: Trained model not found!")
    sys.exit(1)


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

    # Convert extracted features to tensor for PyTorch model
    input_tensor = torch.tensor(extracted_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Step 2: Predict the quantum algorithm using the Neural Network model
    with torch.no_grad():
        predicted_algorithm_idx = nn_model(input_tensor)
        predicted_label = torch.argmax(predicted_algorithm_idx, dim=1).item()

    # Map the predicted label to the algorithm name
    predicted_algorithm = dataset.iloc[predicted_label]["Algorithms"]
    print(f"✅ Selected Quantum Algorithm: {predicted_algorithm}")

    # Step 3: Execute the selected quantum algorithm
    output = execute_algorithm(predicted_algorithm, extracted_features)
    
    # Step 4: Display the result from the quantum algorithm
    print(f"🔹 Solution Output: {output}")


if __name__ == "__main__":
    main()
