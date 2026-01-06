"""
Script: 02_train_model.py
Description: Trains XGBoost models to predict avian haemosporidian infection.
             Includes feature importance analysis using SHAP.
"""

import pandas as pd
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# ================= CONFIGURATION =================
DATA_FILE = 'clean_data.xlsx'
MODEL_DIR = 'models/'  # Directory to save trained models

# Features used for training (Chinese headers matching the dataset)
# English mapping: ['Beak Length', 'Head-Beak Length', 'Wing Length', 'Tail Length', 'Weight']
# Note: '跗跖长mm' is intentionally excluded based on feature selection.
FEATURES = ['喙长mm', '头喙mm', '翼长mm', '尾长mm', '体重g']

TARGET = '感染状态'  # Target: 'Infection Status' (0/1)
SPECIES_COL = '种名' # Column: 'Species Name'
# =================================================

def train_models():
    # Create directory for saving models
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("Loading training data...")
    df = pd.read_excel(DATA_FILE)
    
    # Iterate through each species to train species-specific models
    species_list = df[SPECIES_COL].unique()
    
    for species in species_list:
        # Filter data for the specific species
        species_data = df[df[SPECIES_COL] == species].copy()
        
        # Skip species with insufficient sample size (e.g., n < 40)
        if len(species_data) < 40:
            print(f"Skipping {species}: Insufficient sample size (n={len(species_data)}).")
            continue

        print(f"\nTraining XGBoost model for: {species}...")

        # Prepare features (X) and target (y)
        X = species_data[FEATURES]
        y = species_data[TARGET]

        # Split data: 70% for training, 30% for internal validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # Initialize XGBoost Classifier
        model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate performance
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"  -> Accuracy: {acc:.2f}")

        # Save the trained model (.pkl file)
        model_path = os.path.join(MODEL_DIR, f'{species}_model.pkl')
        joblib.dump(model, model_path)
        print(f"  -> Model saved to: {model_path}")

        # ---------------- SHAP Interpretability Analysis ----------------
        # Explain model predictions using SHAP values
        try:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_train)
            
            # Generate and save SHAP summary plot (Feature Importance)
            plt.figure()
            # Note: feature_names are passed to display original Chinese names in the plot
            shap.summary_plot(shap_values, X_train, feature_names=FEATURES, show=False)
            plt.title(f'SHAP Summary: {species}')
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_DIR, f'{species}_shap.png'))
            plt.close()
        except Exception as e:
            print(f"  Warning: Could not generate SHAP plot for {species}. Error: {e}")

if __name__ == "__main__":
    train_models()
