import pandas as pd
import numpy as np
import re
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score

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

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
log_reg.fit(X_train, y_train)
log_reg_acc = accuracy_score(y_test, log_reg.predict(X_test)) * 100
print(f'Logistic Regression accuracy: {log_reg_acc:.2f}%')

# Train MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=1000, solver='adam', alpha=0.01, random_state=42)
mlp.fit(X_train, y_train)
mlp_acc = accuracy_score(y_test, mlp.predict(X_test)) * 100
print(f'MLP Classifier accuracy: {mlp_acc:.2f}%')

# Train Gradient Boosting Classifier
gbc = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
gbc.fit(X_train, y_train)
gbc_acc = accuracy_score(y_test, gbc.predict(X_test)) * 100
print(f'Gradient Boosting accuracy: {gbc_acc:.2f}%')

# Train Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test)) * 100
print(f'Random Forest accuracy: {rf_acc:.2f}%')

# Select best model
best_model = max([(log_reg, log_reg_acc), (mlp, mlp_acc), (gbc, gbc_acc), (rf, rf_acc)], key=lambda x: x[1])[0]

# Save best model and vectorizer
with open(r"C:\Users\manid\Projects\IQAD\ai_model\models\nlp_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open(r"C:\Users\manid\Projects\IQAD\ai_model\models\vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open(r"C:\Users\manid\Projects\IQAD\ai_model\models\label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Training complete! Best model saved.")

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