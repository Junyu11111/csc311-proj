"""Test prediction script pred.py with given training set"""
import numpy as np
from pred import predict_all
import pandas as pd

filename = "cleaned_data_combined_modified.csv"
df = pd.read_csv(filename)
t = df["Label"]  # Actual classes
y = predict_all(filename)  # Predicted classes

# Compute accuracy
accuracy = np.mean(y == t)

# Print accuracy
print(f"Train Accuracy: {accuracy * 100:.2f}%")
