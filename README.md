# AMHIPT: Avian Morphology-based Haemosporidian Infection Prediction Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)](https://xgboost.readthedocs.io/)

**Official implementation of the paper: [Insert Your Paper Title Here]**

**AMHIPT** is a machine learning pipeline designed to predict haemosporidian infection status in birds using non-invasive morphological measurements (e.g., beak length, weight). Built on **XGBoost** and interpreted via **SHAP**, it provides a cost-effective screening tool for avian researchers.

<p align="center">
  <img src="images/AMHIPT_workflow.png" alt="AMHIPT Workflow" width="85%">
</p>

---

## 📂 Repository Structure

```text
AMHIPT/
├── models/                  # Pre-trained models (.pkl) and Scalers
├── images/
│   └── AMHIPT_workflow.png  # Pipeline flowchart
├── data/
│   ├── example_data.xlsx    # Synthetic dataset for demonstration
│   └── generate_dummy_data.py # Script used to generate dummy data
├── src/
│   ├── 01_data_cleaning.py  # Data preprocessing & English mapping
│   ├── 02_train_model.py    # Model training & SHAP analysis
│   └── 03_predict.py        # Prediction script for independent data
├── requirements.txt         # Python dependencies
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

**Note on Data Availability:**
Due to the proprietary nature of the long-term field dataset, the raw biological data used in this study is **not publicly available** in this repository.

* **`example_data.xlsx`**: A synthetic dataset provided for demonstration purposes. It allows users to test the pipeline structure.
* **Pre-trained Models**: We provide the full `models/` directory containing trained XGBoost models. You can use these to predict *your own* data without needing our raw training data.
* **Access to Raw Data**: Researchers interested in the original dataset may contact the corresponding author upon reasonable request.

---

## 📊 Model Performance & Interpretability

We utilize **SHAP (SHapley Additive exPlanations)** to ensure model transparency. The tool highlights which morphological traits contribute most to the infection probability for each species.

*(You can insert a representative SHAP plot image here)*

## 📝 Citation

If you use AMHIPT in your research, please cite our paper:

> **Ruitian Xu**, **Qingfeng Gan**, **Shiqiong Chuan**, **Xi Huang***, et al. Revise. *A Novel Machine Learning Model for the Prediction of Avian Haemosporidian Infection from Morphological Data*. Avian Research. [DOI Link]

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
