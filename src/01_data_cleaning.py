"""
Script: 01_data_cleaning.py
Description: Preprocesses the raw dataset for the AMHIPT project.
             It handles missing values and formats the data for training.
Note: Column names are preserved in Chinese to match the raw field data.
"""

import pandas as pd
import os

# ================= CONFIGURATION =================
INPUT_FILE = 'raw_data.xlsx'  # Replace with your actual filename
OUTPUT_FILE = 'clean_data.xlsx'

# Feature columns (kept in original Chinese)
# Translation:
# '喙长mm' = Beak Length (mm)
# '头喙mm' = Head-Beak Length (mm)
# '翼长mm' = Wing Length (mm)
# '尾长mm' = Tail Length (mm)
# '体重g'  = Weight (g)
# Note: '跗跖长mm' (Tarsus Length) is excluded from the final model.
FEATURES_TO_CHECK = ['喙长mm', '头喙mm', '翼长mm', '尾长mm', '体重g']
# =================================================

def clean_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

    print("Loading raw data...")
    # Load dataset
    df = pd.read_excel(INPUT_FILE)
    
    print(f"Original dataset size: {len(df)}")

    # Remove rows with missing values in key morphological features
    # This ensures data quality for the machine learning model
    df_clean = df.dropna(subset=FEATURES_TO_CHECK)
    
    # Calculate dropped rows
    dropped_count = len(df) - len(df_clean)
    print(f"Rows dropped due to missing values: {dropped_count}")
    print(f"Cleaned dataset size: {len(df_clean)}")

    # Save the processed data
    df_clean.to_excel(OUTPUT_FILE, index=False)
    print(f"Data processing complete. Saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    clean_data()
