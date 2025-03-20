import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.metrics import accuracy_score

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("IQAD/data/dataset.csv")

df['Keywords'] = df['Keywords'].fillna('')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Expand dataset
expanded_data = []
for _, row in df.iterrows():
    keywords = [clean_text(k) for k in row['Keywords'].split(',')]
    for keyword in keywords:
        expanded_data.append({'Keyword': keyword, 'Algorithm': row['Algorithm']})

expanded_df = pd.DataFrame(expanded_data)

# Feature extraction
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words='english')
X = vectorizer.fit_transform(expanded_df['Keyword'])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(expanded_df['Algorithm'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVC
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_train, y_train)
svc_accuracy = svc.score(X_test, y_test) * 100
print(f'SVC accuracy: {svc_accuracy:.2f}%')

# Save best model and vectorizer
with open("IQAD/ai_model/models/nlp_model.pkl", "wb") as f:
    pickle.dump(svc, f)
with open("IQAD/ai_model/models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("IQAD/ai_model/models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Best model selected: SVC")
print("Training complete! Model saved.")

# Interactive Testing Function
def predict_algorithm(query):
    with open("IQAD/ai_model/models/nlp_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("IQAD/ai_model/models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("IQAD/ai_model/models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    query_cleaned = clean_text(query)
    query_vectorized = vectorizer.transform([query_cleaned])
    prediction = model.predict(query_vectorized)
    predicted_algorithm = label_encoder.inverse_transform(prediction)[0]
    return predicted_algorithm

if __name__ == "__main__":
    while True:
        user_input = input("Enter a query (or type 'exit' to stop): ")
        if user_input.lower() == 'exit':
            break
        result = predict_algorithm(user_input)
        print(f"Predicted Algorithm: {result}")
