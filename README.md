# AMHIPT: Avian Morphology-based Haemosporidian Infection Prediction Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)](https://xgboost.readthedocs.io/)

**Official implementation of the paper: [Insert Your Paper Title Here]**

**AMHIPT** is a machine learning pipeline designed to predict haemosporidian infection status in birds using non-invasive morphological measurements (e.g., beak length, weight). Built on **XGBoost** and interpreted via **SHAP**, it provides a cost-effective screening tool for avian researchers.

---

## 📂 Repository Structure

```text
AMHIPT/
├── models/                  # Saved models (.pkl) and scalers
├── data/                    # Example datasets (ensure anonymity)
├── src/
│   ├── clean_data.py        # Data preprocessing script
│   ├── train_model.py       # Main training script with SHAP analysis
│   └── predict.py           # Prediction script for new samples
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
````

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YourUsername/amhipt.git
   cd amhipt
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### 1. Data Preparation

Input data should be a CSV/Excel file containing standard morphological metrics. Run the cleaning script to format the data and encode infection status.

```bash
python src/clean_data.py
```

*Key Features Used:* Beak Length, Head-Beak Length, Wing Length, Tail Length, Weight.

### 2. Model Training & Interpretation

Train the XGBoost models for specific species. This script will automatically:

* Train separate models for each species (n > 10).
* Generate **SHAP summary plots** to visualize feature importance.
* Save the trained models (`.pkl`) and performance metrics (Log-loss/Accuracy).

```bash
python src/train_model.py
```

*Outputs are saved in a timestamped folder, e.g., `models_20241011/`.*

### 3. Prediction on New Data

Use the trained models to predict infection risks for new, independent samples.

```bash
python src/predict.py
```

*Ensure your input CSV matches the feature columns used during training.*

---

## 📊 Model Performance & Interpretability

We utilize **SHAP (SHapley Additive exPlanations)** to ensure model transparency. The tool highlights which morphological traits contribute most to the infection probability for each species.

*(You can insert a representative SHAP plot image here)*

## 📝 Citation

If you use AMHIPT in your research, please cite our paper:

> **Xi Huang**, [Co-authors], et al. (2024). *Title of Your Paper*. Avian Research. [DOI Link]

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

老师，您先去把代码里的特征 bug 修一下，然后把这两个文件（README.md 和 requirements.txt）加上去，这个仓库就非常完美了！
