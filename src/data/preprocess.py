"""Data preprocessing module for Plant Watering System."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Paths
DATA_PATH = "data/processed/cleaned_data.csv"
PROCESSED_DIR = "data/processed/"
MODELS_DIR = "models/"

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(path):
    """Load dataset from CSV file."""
    return pd.read_csv(path)


def preprocess_data(df):
    """
    Separate features and target, encode target labels.

    Args:
        df: Input DataFrame with all columns.

    Returns:
        tuple: (X, y_encoded, label_encoder)
    """
    target_col = "Plant_Health_Status"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Save label encoder
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print(f"Label Encoder saved to {MODELS_DIR}label_encoder.pkl")

    return X, y_encoded, le


def main():
    """Main preprocessing pipeline."""
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Preprocessing data...")
    X, y, le = preprocess_data(df)

    # Stratified split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Identify columns to scale
    exclude_cols = ["Timestamp", "Plant_ID"]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scale_cols = [col for col in numeric_cols if col not in exclude_cols]

    print(f"Columns to scale: {scale_cols}")
    print(f"Columns excluded from scaling: {exclude_cols}")

    # Feature Scaling
    print("Scaling features...")
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

    # Save scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print(f"Scaler saved to {MODELS_DIR}scaler.pkl")

    # Save processed data
    print("Saving processed data...")
    X_train_scaled.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    pd.DataFrame(y_train, columns=["Plant_Health_Status"]).to_csv(
        os.path.join(PROCESSED_DIR, "y_train.csv"), index=False
    )
    pd.DataFrame(y_test, columns=["Plant_Health_Status"]).to_csv(
        os.path.join(PROCESSED_DIR, "y_test.csv"), index=False
    )

    print("Data processing complete!")


if __name__ == "__main__":
    main()
