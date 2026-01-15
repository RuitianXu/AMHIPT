"""
Script: 02_train_model.py
Description: Trains XGBoost models for avian haemosporidian infection prediction.
             Outputs: Trained models (.pkl), Scalers, Summary Excel, and Performance Plots.
"""

import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ================= CONFIGURATION =================
DATA_FILE = 'clean_data.xlsx'

# Features used for training (Chinese headers matching the dataset)
# English mapping: ['Beak_Length', 'Head_Beak_Length', 'Wing_Length', 'Tail_Length', 'Weight']
# Note: '跗跖长mm' is intentionally excluded based on feature selection.
FEATURES = ['喙长mm', '头喙mm', '翼长mm', '尾长mm', '体重g']
TARGET = '感染状态'  # Target: 'Infection Status' (0/1)
SPECIES_COL = '种名' # Column: 'Species Name'

SAMPLE_SIZE_THRESHOLD = 60  # Minimum samples required to train

# Plotting settings for Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# =================================================

def setup_output_dir():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'models_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, timestamp

def train_and_evaluate():
    output_dir, current_time = setup_output_dir()
    summary_path = os.path.join(output_dir, f'Summary_{current_time}.xlsx')

    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    df = pd.read_excel(DATA_FILE)
    species_list = df[SPECIES_COL].unique()
    
    results_storage = {}
    logloss_history = {}
    accuracy_list = []

    print(f"Starting training for {len(species_list)} species...")

    for species in species_list:
        sub_df = df[df[SPECIES_COL] == species].dropna(subset=FEATURES + [TARGET])
        
        if len(sub_df) < SAMPLE_SIZE_THRESHOLD:
            continue

        print(f"Processing: {species} (n={len(sub_df)})")

        X = sub_df[FEATURES]
        y = sub_df[TARGET]

        # Split 70/30 stratified
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Standardization (Crucial: save scaler for inference)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train XGBoost
        model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=200,
            random_state=42
        )
        
        eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
        model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)

        # Evaluation
        y_pred = model.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_pred)
        
        # Save artifacts
        joblib.dump(model, os.path.join(output_dir, f'{species}_model.pkl'))
        joblib.dump(scaler, os.path.join(output_dir, f'{species}_scaler.pkl'))

        # Store results
        logloss_history[species] = model.evals_result()['validation_1']['logloss']
        accuracy_list.append((species, acc))
        
        feat_imp = pd.DataFrame({
            'Feature': FEATURES,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        results_storage[species] = {
            'accuracy': acc,
            'infected': y.sum(),
            'clean': len(y) - y.sum(),
            'report': classification_report(y_val, y_pred),
            'importance': feat_imp
        }

    # --- Generate Outputs ---
    if not results_storage:
        print("No models trained.")
        return

    # 1. Excel Report
    with pd.ExcelWriter(summary_path, engine='xlsxwriter') as writer:
        # Summary Sheet
        pd.DataFrame({
            'Species': list(results_storage.keys()),
            'Accuracy': [v['accuracy'] for v in results_storage.values()],
            'Infected_N': [v['infected'] for v in results_storage.values()],
            'Uninfected_N': [v['clean'] for v in results_storage.values()]
        }).to_excel(writer, sheet_name='Overview', index=False)

        # Detail Sheets
        for sp, data in results_storage.items():
            sheet_name = sp.replace('/', '_')[:25]
            pd.DataFrame([data['report'].splitlines()]).T.to_excel(writer, sheet_name=f'{sheet_name}_Rpt', header=False)
            data['importance'].to_excel(writer, sheet_name=f'{sheet_name}_Imp', index=False)

    # 2. Logloss Plot
    plt.figure(figsize=(10, 6))
    for sp, loss in logloss_history.items():
        plt.plot(loss, label=sp, alpha=0.8)
    plt.title('Validation Logloss per Species')
    plt.xlabel('Trees')
    plt.ylabel('Logloss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'logloss_curves.png'), dpi=300)
    plt.close()

    # 3. Accuracy Plot
    if accuracy_list:
        accuracy_list.sort(key=lambda x: x[1])
        names, accs = zip(*accuracy_list)
        plt.figure(figsize=(8, len(names)*0.5 + 2))
        bars = plt.barh(names, accs, color='#9BAFBA', edgecolor='#4A6572') # Professional colors
        plt.xlim(0, 1.0)
        plt.title('Model Accuracy by Species')
        plt.bar_label(bars, fmt='%.2f', padding=3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300)
        plt.close()

    print(f"\nDone. Results saved to: {output_dir}")

if __name__ == "__main__":
    train_and_evaluate()
