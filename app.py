from flask import Flask, render_template, request, jsonify
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai_model.models.nlp_model import clean_text
from algorithms import shor_algorithm
import re

app = Flask(__name__)

# Load saved models
with open("C:\\Users\\manid\\Projects\\IQAD\\ai_model\\models\\nlp_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("C:\\Users\\manid\\Projects\\IQAD\\ai_model\\models\\vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("C:\\Users\\manid\\Projects\\IQAD\\ai_model\\models\\label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def predict_algorithm(query):
    query_cleaned = clean_text(query)
    query_vectorized = vectorizer.transform([query_cleaned])
    prediction = model.predict(query_vectorized)
    predicted_algorithm = label_encoder.inverse_transform(prediction)[0]
    result = f"Running {predicted_algorithm} on input: {query}"  # Placeholder for actual function
    return predicted_algorithm, result

@app.route("/")
def home():
    return render_template("index.html")

def is_valid_input(query):
    return query.isdigit() or any(c.isalpha() for c in query)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Empty query"}), 400
    user_query = data.get("query", "").strip()

    if not is_valid_input(user_query):
        return jsonify({"response": "This is invalid. Please enter a valid query."})

    predicted_algorithm, result = predict_algorithm(query)

    # Algorithm implementation
    if predicted_algorithm == "Shor's Algorithm":
        match = re.search(r"\b\d+\b", query)
        if match:
            number = match.group(0)
            factors = shor_algorithm.factorize(int(number))
            factor_statement = f"Factors of {number}: {factors}"

    else:
        factor_statement = ""


    return jsonify({"predicted_algorithm": predicted_algorithm, "result": result, "factors":factor_statement})


if __name__ == "__main__":
    app.run(debug=True)
