import pandas as pd
import numpy as np
import re
import pickle
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv(r"C:\Users\manid\Projects\IQAD\data\dataset.csv")
df['Keywords'] = df['Keywords'].fillna('')

# Initialize stemmer
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9 +\-*]', '', text)  # Keep mathematical operators
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Expand dataset
expanded_data = []
for _, row in df.iterrows():
    keywords = [clean_text(k) for k in row['Keywords'].split(',')]
    for keyword in keywords:
        expanded_data.append({'Keyword': keyword, 'Algorithm': row['Algorithm']})

expanded_df = pd.DataFrame(expanded_data)

# Feature extraction
vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=7000)  # Increased max_features
X = vectorizer.fit_transform(expanded_df['Keyword'])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(expanded_df['Algorithm'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,100), max_iter=1000, solver='adam', alpha=0.01, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute scores
    acc = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100

    # Store scores
    scores[name] = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

# Convert scores to DataFrame for plotting
scores_df = pd.DataFrame(scores).T

# Plot bar chart comparison
plt.figure(figsize=(12, 6))
ax = scores_df.plot(kind="bar", figsize=(12, 6), colormap="viridis", edgecolor='black')

# Annotate bars with values
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=10, padding=3)

# plt.title("Comparison of Model Performance", fontsize=14)
# plt.ylabel("Score (%)", fontsize=12)
# plt.xticks(rotation=45, fontsize=10)
# plt.legend(title="Metrics", fontsize=10)
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.show()

# Select best model based on Accuracy
best_model_name = max(scores, key=lambda x: scores[x]["Accuracy"])
best_model = models[best_model_name]

# Save best model and vectorizer
with open(r"C:\Users\manid\Projects\IQAD\ai_model\models\nlp_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open(r"C:\Users\manid\Projects\IQAD\ai_model\models\vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open(r"C:\Users\manid\Projects\IQAD\ai_model\models\label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print(f"Training complete! Best model ({best_model_name}) saved.")

# Define quantum algorithm execution function
def run_algorithm(algorithm, query):
    if algorithm == "Shor's Algorithm":
        return f"Executing Shor's Algorithm on {query}... Factorization complete."
    elif algorithm == "Grover's Algorithm":
        return f"Executing Grover's Algorithm on {query}... Optimal solution found."
    elif algorithm == "Simon's Algorithm":
        return f"Executing Simon's Algorithm on {query}... Hidden pattern discovered."
    elif algorithm == "Quantum Fourier Transform":
        return f"Executing Quantum Fourier Transform on {query}... Frequency domain representation obtained."
    else:
        return f"Algorithm {algorithm} not found."

# Interactive Testing Function
def predict_algorithm(query):
    with open(r"C:\Users\manid\Projects\IQAD\ai_model\models\nlp_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(r"C:\Users\manid\Projects\IQAD\ai_model\models\vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(r"C:\Users\manid\Projects\IQAD\ai_model\models\label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    query_cleaned = clean_text(query)
    query_vectorized = vectorizer.transform([query_cleaned])
    prediction = model.predict(query_vectorized)
    predicted_algorithm = label_encoder.inverse_transform(prediction)[0]

    # Call run_algorithm() correctly
    result = run_algorithm(predicted_algorithm, query)  

    return predicted_algorithm, result
