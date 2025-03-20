import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai_model.models.nlp_model import predict_algorithm

def main():
    print("Quantum Algorithm Predictor. Type 'exit' to quit.")
    while True:
        user_input = input("Enter a query: ")
        if user_input.lower() == 'exit':
            break
        predicted_algorithm, result = predict_algorithm(user_input)
        print(f"Predicted Algorithm: {predicted_algorithm}")
        print(f"Result: {result}")

if __name__ == "__main__":
    main()