"""
Script: 03_predict.py
Description: Loads pre-trained models to predict infection status on new, independent datasets.
             This script validates the AMHIPT tool on unseen data (e.g., 2023-2024 samples).
"""

import pandas as pd
import joblib
import os

# ================= CONFIGURATION =================
# New dataset for prediction (Independent Test Set)
NEW_DATA_FILE = 'new_data_2024.xlsx'
MODEL_DIR = 'models/'
OUTPUT_FILE = 'prediction_results.xlsx'

# Must match the training features exactly!
# FEATURES =['Beak_Length', 'Head_Bill_Length', 'Wing_Length', 'Tail_Length', 'Weight']
FEATURES = ['喙长mm', '头喙mm', '翼长mm', '尾长mm', '体重g']
SPECIES_COL = '种名'
# =================================================

def predict_new_data():
    if not os.path.exists(NEW_DATA_FILE):
        print(f"Error: Prediction file '{NEW_DATA_FILE}' not found.")
        return

    print("Loading new dataset...")
    df_new = pd.read_excel(NEW_DATA_FILE)
    
    # Initialize columns for predictions
    df_new['Predicted_Status'] = None
    df_new['Infection_Probability'] = None

    unique_species = df_new[SPECIES_COL].unique()

    for species in unique_species:
        model_path = os.path.join(MODEL_DIR, f'{species}_model.pkl')
        
        # Check if a trained model exists for this species
        if not os.path.exists(model_path):
            print(f"Warning: No model found for {species}. Skipping prediction.")
            continue
            
        print(f"Predicting infection status for: {species}...")
        
        # Load the pre-trained model
        model = joblib.load(model_path)
        
        # Extract data indices for the current species
        indices = df_new[df_new[SPECIES_COL] == species].index
        
        # Select features for prediction
        # (XGBoost requires the input columns to match training exactly)
        X_sub = df_new.loc[indices, FEATURES]
        
        # Perform prediction
        preds = model.predict(X_sub)
        probs = model.predict_proba(X_sub)[:, 1] # Probability of Class 1 (Infected)
        
        # Store results back to the dataframe
        df_new.loc[indices, 'Predicted_Status'] = preds
        df_new.loc[indices, 'Infection_Probability'] = probs

    # Save final results
    df_new.to_excel(OUTPUT_FILE, index=False)
    print(f"\nPrediction complete. Results saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    predict_new_data()
