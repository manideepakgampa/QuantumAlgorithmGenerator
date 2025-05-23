from flask import Flask, render_template, request, jsonify
import pickle
import sys
import os
import subprocess
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai_model.models.nlp_model import clean_text

app = Flask(__name__)

# Load saved models
with open("ai_model\\models\\nlp_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("ai_model\\models\\vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("ai_model\\models\\label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def predict_algorithm(query):
    query_cleaned = clean_text(query)
    query_vectorized = vectorizer.transform([query_cleaned])
    prediction = model.predict(query_vectorized)
    predicted_algorithm = label_encoder.inverse_transform(prediction)[0]
    result = f"Running {predicted_algorithm} on input: {query}"
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

    user_query = query.strip()

    if not is_valid_input(user_query):
        return jsonify({"response": "This is invalid. Please enter a valid query."})

    predicted_algorithm, result = predict_algorithm(user_query)

    factor_statement = ""

    if predicted_algorithm == "Shor's Algorithm":
        try:
            subprocess.run(["python", "algorithms/shor_algorithm.py"])

            output_path = "shor_output.txt"
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    content = f.read()
                    if ":" in content:
                        number, factors = content.strip().split(":")
                        factor_statement = f"Shor’s Algorithm Executed Successfully.<br> Factors of {number} are: {factors}"
                    else:
                        factor_statement = "Shor’s Algorithm Executed, but no valid output found."
                os.remove(output_path)
            else:
                factor_statement = "Shor’s Algorithm Executed, but result file not found."

        except Exception as e:
            factor_statement = f"Error executing GUI: {str(e)}"
    elif predicted_algorithm == "Grover's Algorithm":
        try:
            grover_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "algorithms", "grover_algorithm.py"))
            print(f"🚀 Launching Grover GUI at: {grover_path}")

            subprocess.run([sys.executable, grover_path])

            output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "grover_output.txt"))
            print(f"🔍 Flask is looking for Grover output at: {output_path}")

            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    content = f.read().strip()
                    if content:
                        factor_statement = f"Grover’s Algorithm Executed Successfully.<br> Measured state: |{int(content):0{len(content)}b}>"
                    else:
                        factor_statement = "Grover’s Algorithm Executed, but no valid output found."
                os.remove(output_path)
            else:
                factor_statement = "Grover’s Algorithm Executed, but result file not found."

        except Exception as e:
            factor_statement = f"Error executing Grover GUI: {str(e)}"
    elif predicted_algorithm == "Quantum Fourier Transform":
        try:
            qft_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "algorithms", "qft.py"))
            print(f"🚀 Launching QFT GUI at: {qft_path}")

            subprocess.run([sys.executable, qft_path])

            output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "qft_output.txt"))
            print(f"🔍 Flask is looking for QFT output at: {output_path}")

            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    content = f.read().strip()
                    if content:
                        factor_statement = f"QFT Executed Successfully.<br> Measured state: |{content}>"
                    else:
                        factor_statement = "QFT Executed, but no valid output found."
                os.remove(output_path)
            else:
                factor_statement = "QFT Executed, but result file not found."

        except Exception as e:
            factor_statement = f"Error executing QFT GUI: {str(e)}"
    elif predicted_algorithm == "Quantum Teleportation":
        try:
            teleport_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "algorithms", "quantum_teleportation.py"))
            print(f"🚀 Launching Quantum Teleportation GUI at: {teleport_path}")

            subprocess.run([sys.executable, teleport_path])

            output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "teleportation_output.txt"))
            print(f"🔍 Flask is looking for teleportation output at: {output_path}")

            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    content = f.read().strip()
                    if content:
                        factor_statement = f"Quantum Teleportation Executed Successfully.<br> Bob's qubit state: |{content}>"
                    else:
                        factor_statement = "Quantum Teleportation Executed, but no valid output found."
                os.remove(output_path)
            else:
                factor_statement = "Quantum Teleportation Executed, but result file not found."

        except Exception as e:
            factor_statement = f"Error executing Quantum Teleportation GUI: {str(e)}"
    elif predicted_algorithm == "Simon's Algorithm":
        try:
            simon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "algorithms", "simon.py"))
            print(f"🚀 Launching Simon's Algorithm GUI at: {simon_path}")

            subprocess.run([sys.executable, simon_path])

            output_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"simons_output.txt"))
            print(f"🔍 Flask is looking for Simon output at: {output_path}")

            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    content = f.read().strip()
                    if content:
                        factor_statement = f"Simon's Algorithm Executed Successfully.<br> Hidden string s = {content}"
                    else:
                        factor_statement = "Simon's Algorithm Executed, but no valid output found."
                os.remove(output_path)
            else:
                factor_statement = "Simon's Algorithm Executed, but result file not found."

        except Exception as e:
            factor_statement = f"Error executing Simon's Algorithm GUI: {str(e)}"


    else:
        factor_statement = "Algorithm currently not supported with GUI."

    return jsonify({
        "predicted_algorithm": predicted_algorithm,
        "result": result,
        "factors": factor_statement
    })

if __name__ == "__main__":
    app.run(debug=True)
