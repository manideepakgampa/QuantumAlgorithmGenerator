import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import DataLoader
from neural_network import FeedforwardNN, initialize_model, train_model

class NLPModel:
    def __init__(self, dataset_path='IQAD/ai_model/data/dataset.csv'):
        # Load the dataset
        self.df = pd.read_csv(dataset_path)
        
        # Extract keywords and algorithms
        self.keywords = self.df['Keywords'].values
        self.algorithms = self.df['Algorithm'].values
        
        # Encode the labels (Algorithm)
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.algorithms)
        
        # Split the data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.keywords, self.encoded_labels, test_size=0.2, random_state=42
        )
        
        # Vectorize keywords using TF-IDF
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.X_train_tfidf = self.vectorizer.fit_transform(self.X_train).toarray()
        self.X_test_tfidf = self.vectorizer.transform(self.X_test).toarray()

        # Convert to PyTorch tensors
        self.X_train_tensor = torch.tensor(self.X_train_tfidf, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(self.X_test_tfidf, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(self.y_train, dtype=torch.long)
        self.y_test_tensor = torch.tensor(self.y_test, dtype=torch.long)
        
        # Create Dataset and DataLoader
        self.train_dataset = torch.utils.data.TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.test_dataset = torch.utils.data.TensorDataset(self.X_test_tensor, self.y_test_tensor)
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32)

    def train(self, input_size, hidden_size, output_size, num_epochs=10):
        model, criterion, optimizer = initialize_model(input_size, hidden_size, output_size)
        train_model(model, self.train_loader, criterion, optimizer, num_epochs)
        return model

    def get_encoded_labels(self):
        return self.encoded_labels
    
    def get_label_encoder(self):
        return self.label_encoder
