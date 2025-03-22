import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
dataset_path = "C:\Users\Noel.NOELKING\Desk-files\IQAD\quantum\IQAD\data\quantum_queries_dataset.csv"  # Update path if needed
df = pd.read_csv(dataset_path)

# Encode categorical labels (problem types -> algorithm names)
categorical_columns = ["Query", "Algorithm"]  # Update based on your dataset

# Encode categorical columns using LabelEncoder
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Separate features and labels
X = df.drop(columns=["Algorithm"]).values  # Features
y = df["Algorithm"].values  # Labels

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors (No reshaping issue now)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Debugging: Check feature size
print("X_train shape:", X_train.shape)  # Ensure correct feature count

# Define Neural Network Model
class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Model parameters
input_size = X_train.shape[1]  # Dynamically get correct feature count
hidden_size = 100  
output_size = len(np.unique(y))  

# Debugging: Ensure input size matches model
print("Model input size:", input_size)

# Initialize model, loss function, and optimizer
model = FeedforwardNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Save trained model
model_path = "iqad_trained_model.pth"
torch.save(model.state_dict(), model_path)

print(f"Model saved as {model_path}")
