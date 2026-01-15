"""
Script: 02_model_train_multiclass.py
Description: Trains XGBoost models for predicting Parasite Genus (Multi-class).
             Filters: Specific bird species, Non-empty genus records.
             Outputs: Trained models (.pkl), Scalers, Encoders, Summary Excel.
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

warnings.filterwarnings('ignore', category=UserWarning)

# ================= CONFIGURATION =================
DATA_FILE = 'final_clean_species_data.csv' # Your data file

# Features (Input)
FEATURES = ['喙长mm', '头喙mm', '翼长mm', '尾长mm', '体重g']

# Target (Output): Now predicting the Genus
TARGET = '属'

# Species Column
SPECIES_COL = '物种'

# The 8 specific species required
# For your data, you may need to adjust the species names (scientific names) to match your data
TARGET_SPECIES_LIST = [
    "银喉长尾山雀",
    "黄喉鹀",
    "大山雀",
    "沼泽山雀",
    "红胁蓝尾鸲",
    "黄腰柳莺",
    "褐头山雀",
    "黄腹山雀"
]

# Minimum samples required to train (Lowered slightly as we filtered out healthy birds)
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
    if os.path.exists(DATA_FILE):
        if DATA_FILE.endswith('.csv'):
            df = pd.read_csv(DATA_FILE, encoding='gbk')
        else:
            df = pd.read_excel(DATA_FILE)
    else:
        print(f"Error: {DATA_FILE} not found.")
        return

    df.columns = df.columns.str.strip()
    
    if SPECIES_COL not in df.columns:
        print(f"Error: Can't find column'{SPECIES_COL}', the column names are: {df.columns.tolist()}")
        return

    # 2. Filter Data
    df[SPECIES_COL] = df[SPECIES_COL].astype(str).str.strip()
    
    # 3. Filter Target Species
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df[df[TARGET] != '']
    df = df[df[TARGET] != 'nan']

    # 4. Filter Specific Species
    df = df[df[SPECIES_COL].isin(TARGET_SPECIES_LIST)]

    print(f"Data counts (Infected & Filtered): {len(df)}")
    
    results_storage = {}
    logloss_history = {}
    accuracy_list = []

    species_to_process = sorted(list(set(df[SPECIES_COL].unique())))

    print(f"\nAbout to process {len(species_to_process)} species: {species_to_process}")

    for species in species_to_process:
        sub_df = df[df[SPECIES_COL] == species].dropna(subset=FEATURES + [TARGET])
        
        if len(sub_df) < SAMPLE_SIZE_THRESHOLD:
            print(f"Skipping {species}: Lack of  (n={len(sub_df)})")
            continue

        X = sub_df[FEATURES]
        y_raw = sub_df[TARGET]

        if y_raw.nunique() < 2:
            print(f"Skipping {species}: Find only one parasite ({y_raw.unique()[0]}), unable to train a model")
            continue

        print(f"Processing: {species} (n={len(sub_df)}, Parasite Genera={y_raw.nunique()})")

        # --- PREPROCESSING ---
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_raw)
        num_classes = len(le.classes_)

        # split data
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
            )
        except ValueError:
            # If there are only two classes, stratify will fail
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42
            )

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
        
        y_val_labels = le.inverse_transform(y_val)
        y_pred_labels = le.inverse_transform(y_pred)
        
        acc = accuracy_score(y_val, y_pred)
        
        # --- SAVE ---
        joblib.dump(model, os.path.join(output_dir, f'{species}_model.pkl'))
        joblib.dump(scaler, os.path.join(output_dir, f'{species}_scaler.pkl'))
        joblib.dump(le, os.path.join(output_dir, f'{species}_label_encoder.pkl'))

        if'validation_1' in model.evals_result():
            logloss_history[species] = model.evals_result()['validation_1']['mlogloss']
        
        accuracy_list.append((species, acc))
        
        feat_imp = pd.DataFrame({
            'Feature': FEATURES,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        cls_report = classification_report(y_val_labels, y_pred_labels, zero_division=0)

        results_storage[species] = {
            'accuracy': acc,
            'samples': len(sub_df),
            'classes': ", ".join(map(str, le.classes_)),
            'report': cls_report,
            'importance': feat_imp
        }

    # --- OUTPUT ---
    if not results_storage:
        print("No results to save")
        return

    with pd.ExcelWriter(summary_path, engine='xlsxwriter') as writer:
        pd.DataFrame({
            'Species': list(results_storage.keys()),
            'Accuracy': [v['accuracy'] for v in results_storage.values()],
            'Total_Samples': [v['samples'] for v in results_storage.values()],
            'Genera': [v['classes'] for v in results_storage.values()]
        }).to_excel(writer, sheet_name='Overview', index=False)

        for sp, data in results_storage.items():
            sheet_name = sp.replace('/', '_')[:20]
            report_df = pd.DataFrame([x.split() for x in data['report'].split('\n') if x], index=None)
            report_df.to_excel(writer, sheet_name=f'{sheet_name}_Rpt', header=False, index=False)
            data['importance'].to_excel(writer, sheet_name=f'{sheet_name}_Imp', index=False)

    if logloss_history:
        plt.figure(figsize=(10, 6))
        for sp, loss in logloss_history.items():
            plt.plot(loss, label=sp, alpha=0.8)
        plt.title('Validation mLogloss')
        plt.xlabel('Trees')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'logloss_curves.png'), dpi=300)
        plt.close()

    if accuracy_list:
        accuracy_list.sort(key=lambda x: x[1])
        names, accs = zip(*accuracy_list)
        plt.figure(figsize=(8, max(4, len(names)*0.6)))
        bars = plt.barh(names, accs, color='#FF6F61', edgecolor='#A13D2D')
        plt.xlim(0, 1.0)
        plt.title('Genus Prediction Accuracy')
        plt.bar_label(bars, fmt='%.2f', padding=3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300)
        plt.close()

    print(f"\nDone. Results saved to: {output_dir}")

if __name__ == "__main__":
    train_and_evaluate()







