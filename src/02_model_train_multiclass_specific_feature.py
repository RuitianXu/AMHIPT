"""
Script: 02_model_train_multiclass.py
Description: Trains XGBoost models for predicting Parasite Genus (Multi-class).
             Features are selected dynamically per species based on previous analysis.
             Outputs: Trained models (.pkl), Scalers, Encoders, feature lists, and Summary Excel.
"""

import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import os
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ================= CONFIGURATION =================
DATA_FILE = 'final_clean_species_data.csv'

# Target (Output): Parasite Genus
TARGET = '属'
# Species Column Header in CSV
SPECIES_COL = '物种'

# Feature mapping based on Table 1 analysis
# Keys are Chinese names (matching CSV), values are the selected features
SPECIES_FEATURE_MAP = {
    "黄腹山雀": ['喙长mm', '头喙mm', '体重g'],                    # Pardaliparus venustulus
    "银喉长尾山雀": ['喙长mm', '头喙mm', '翼长mm', '尾长mm', '体重g'], # Aegithalos glaucogularis
    "红胁蓝尾鸲": ['喙长mm', '头喙mm', '翼长mm', '尾长mm', '体重g'],   # Tarsiger cyanurus
    "褐头山雀": ['喙长mm', '头喙mm', '翼长mm', '尾长mm', '体重g'],     # Poecile montanus
    "大山雀": ['喙长mm', '头喙mm', '翼长mm', '尾长mm', '体重g'],       # Parus minor
    "沼泽山雀": ['喙长mm', '头喙mm', '尾长mm'],                     # Poecile palustris
    "黄喉鹀": ['头喙mm', '翼长mm', '体重g'],                        # Emberiza elegans
    "黄腰柳莺": ['喙长mm', '头喙mm', '翼长mm', '尾长mm', '体重g']      # Phylloscopus proregulus
}

TARGET_SPECIES_LIST = list(SPECIES_FEATURE_MAP.keys())
SAMPLE_SIZE_THRESHOLD = 20 

# Plotting settings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# =================================================

def setup_output_dir():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'models_multiclass_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, timestamp

def train_and_evaluate():
    output_dir, current_time = setup_output_dir()
    summary_path = os.path.join(output_dir, f'Summary_{current_time}.xlsx')

    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    try:
        # Try GB18030 first (superset of GBK, handles more Chinese chars)
        df = pd.read_csv(DATA_FILE, encoding='gb18030')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(DATA_FILE, encoding='utf-8')
        except Exception:
            df = pd.read_excel(DATA_FILE)

    # Clean headers
    df.columns = df.columns.str.strip()
    
    if SPECIES_COL not in df.columns:
        print(f"Error: Column '{SPECIES_COL}' not found. Available columns: {df.columns.tolist()}")
        return

    # 2. Data Cleaning
    # Clean species names (remove invisible chars)
    df[SPECIES_COL] = df[SPECIES_COL].astype(str).str.strip()
    
    # Filter: Only infected records (Genus is not empty)
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df[df[TARGET] != '']
    df = df[df[TARGET] != 'nan']

    # Filter: Only target species
    df = df[df[SPECIES_COL].isin(TARGET_SPECIES_LIST)]

    print(f"Total infected records for target species: {len(df)}")
    
    results_storage = {}
    accuracy_list = []

    # Get sorted list of species actually present in data
    species_to_process = sorted(list(set(df[SPECIES_COL].unique())))

    print(f"\nProcessing {len(species_to_process)} species...")

    for species in species_to_process:
        # Get specific features for this species
        current_features = SPECIES_FEATURE_MAP.get(species)
        
        if not current_features:
            print(f"Skipping {species}: No features defined.")
            continue

        # Drop rows where specific features are missing
        sub_df = df[df[SPECIES_COL] == species].dropna(subset=current_features + [TARGET])
        
        if len(sub_df) < SAMPLE_SIZE_THRESHOLD:
            print(f"Skipping {species}: Insufficient samples (n={len(sub_df)})")
            continue

        X = sub_df[current_features]
        y_raw = sub_df[TARGET]

        # Ensure multi-class training is possible
        if y_raw.nunique() < 2:
            print(f"Skipping {species}: Only 1 parasite genus found ({y_raw.unique()[0]}).")
            continue

        print(f"Training: {species} (n={len(sub_df)}, Features={len(current_features)})")

        # --- PREPROCESSING ---
        # Encode Target (String -> Int)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_raw)
        num_classes = len(le.classes_)

        # Split Data
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
            )
        except ValueError:
            # Fallback to random split if stratification fails (rare classes)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42
            )

        # Scale Features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # --- TRAINING ---
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            eval_metric='mlogloss',
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        
        eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
        model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)

        # --- EVALUATION ---
        y_pred = model.predict(X_val_scaled)
        
        # Decode predictions
        y_val_labels = le.inverse_transform(y_val)
        y_pred_labels = le.inverse_transform(y_pred)
        
        acc = accuracy_score(y_val, y_pred)
        
        # --- SAVE ARTIFACTS ---
        joblib.dump(model, os.path.join(output_dir, f'{species}_model.pkl'))
        joblib.dump(scaler, os.path.join(output_dir, f'{species}_scaler.pkl'))
        joblib.dump(le, os.path.join(output_dir, f'{species}_label_encoder.pkl'))
        joblib.dump(current_features, os.path.join(output_dir, f'{species}_features.pkl'))
        
        accuracy_list.append((species, acc))
        
        # Calculate feature importance
        feat_imp = pd.DataFrame({
            'Feature': current_features,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        cls_report = classification_report(y_val_labels, y_pred_labels, zero_division=0)

        results_storage[species] = {
            'accuracy': acc,
            'samples': len(sub_df),
            'features': ", ".join(current_features),
            'classes': ", ".join(map(str, le.classes_)),
            'report': cls_report,
            'importance': feat_imp
        }

    # --- OUTPUT GENERATION ---
    if not results_storage:
        print("No models trained.")
        return

    # Save Summary Excel
    with pd.ExcelWriter(summary_path, engine='xlsxwriter') as writer:
        # Overview Sheet
        pd.DataFrame({
            'Species': list(results_storage.keys()),
            'Accuracy': [v['accuracy'] for v in results_storage.values()],
            'Samples': [v['samples'] for v in results_storage.values()],
            'Features_Used': [v['features'] for v in results_storage.values()]
        }).to_excel(writer, sheet_name='Overview', index=False)

        # Detail Sheets
        for sp, data in results_storage.items():
            sheet_name = sp.replace('/', '_')[:20]
            # Format report for Excel
            report_df = pd.DataFrame([x.split() for x in data['report'].split('\n') if x])
            report_df.to_excel(writer, sheet_name=f'{sheet_name}_Rpt', header=False, index=False)
            data['importance'].to_excel(writer, sheet_name=f'{sheet_name}_Imp', index=False)

    # Generate Accuracy Plot
    if accuracy_list:
        accuracy_list.sort(key=lambda x: x[1])
        names, accs = zip(*accuracy_list)
        plt.figure(figsize=(8, max(4, len(names)*0.6)))
        bars = plt.barh(names, accs, color='#5DADE2', edgecolor='#2874A6')
        plt.xlim(0, 1.0)
        plt.title('Genus Prediction Accuracy (Selected Features)')
        plt.bar_label(bars, fmt='%.2f', padding=3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300)
        plt.close()

    print(f"\nDone. Results and models saved to: {output_dir}")

if __name__ == "__main__":
    train_and_evaluate()
