# AMHIPT: Avian Morphology-based Haemosporidian Infection Prediction Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)](https://xgboost.readthedocs.io/)

**Official implementation of the paper: A Novel Machine Learning Model for the Prediction of Avian Haemosporidian Infection from Morphological Data**

**AMHIPT** is a machine learning pipeline designed to predict haemosporidian infection status in birds using non-invasive morphological measurements (e.g., beak length, weight). Built on **XGBoost with species-specific feature optimization**, it provides a cost-effective screening tool for avian researchers.

<p align="center">
  <img src="image/pipline_for_xgb.png" alt="AMHIPT Workflow" width="85%">
</p>

---

## 📂 Repository Structure

```text
AMHIPT/
├── models/                  # Pre-trained models (.pkl) and Scalers
├── image/
│   └── pipline_for_xgb.png  # Pipeline flowchart
├── data/
│   ├── example_data.xlsx    # Synthetic dataset for demonstration
│   └── generate_dummy_data.py # Script used to generate dummy data
├── src/
│   ├── 01_data_cleaning.py  # Data preprocessing & English mapping
│   ├── 02_model_train.py    # Binary classification: Infection prediction
│   ├── 02_model_train_multiclass_specific_feature.py # Multi-class: Parasite genus prediction
│   └── 03_predict.py        # Prediction script for independent data
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
````

---

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

---

## 🚀 Usage

### 1. Data Preparation

Input data should be a CSV/Excel file containing standard morphological metrics. Run the cleaning script to format the data and encode infection status.

```bash
python src/01_data_cleaning.py
```

*Key Features Used:* Beak Length, Head-Beak Length, Wing Length, Tail Length, Weight.

### 2. Model Training

AMHIPT supports two types of prediction tasks:

#### **A. Binary Classification (Infection vs. Non-infection)**

Train XGBoost models to predict whether a bird is infected with haemosporidian parasites.

```bash
python src/02_model_train.py
```

**Outputs:**

* Trained models (`.pkl`) for each species
* SHAP summary plots visualizing feature importance
* Performance metrics (Accuracy, Log-loss)

#### **B. Multi-class Classification (Parasite Genus Prediction)**

For infected birds, predict the specific parasite genus (*Haemoproteus*, *Plasmodium*, *Leucocytozoon*).

```bash
python src/02_model_train_multiclass_specific_feature.py
```

**Key Features:**

* Uses **species-specific feature subsets** optimized through iterative training
* Different species achieve optimal accuracy with different feature combinations
* Automatically handles label encoding for parasite genera

**Species-Specific Feature Selection:**

Our three-round training process revealed that optimal feature combinations vary by species, reflecting the diverse morphological responses to parasitic infection across taxa:

| **Species**                | **Sample Size** | **Infected** | **Selected Features**                                                  | **Accuracy** |
| -------------------------- | --------------- | ------------ | ---------------------------------------------------------------------- | ------------ |
| *Pardaliparus venustulus*  | 145             | 103          | Beak length / Head-beak length / Body mass                             | 0.841        |
| *Aegithalos glaucogularis* | 72              | 25           | Beak length / Head-beak length / Wing length / Tail length / Body mass | 0.818        |
| *Tarsiger cyanurus*        | 67              | 55           | Beak length / Head-beak length / Wing length / Tail length / Body mass | 0.810        |
| *Poecile montanus*         | 79              | 64           | Beak length / Head-beak length / Wing length / Tail length / Body mass | 0.792        |
| *Parus minor*              | 61              | 52           | Beak length / Head-beak length / Wing length / Tail length / Body mass | 0.789        |
| *Poecile palustris*        | 64              | 53           | Beak length / Head-beak length / Tail length                           | 0.750        |
| *Emberiza elegans*         | 79              | 39           | Head-beak length / Wing length / Body mass                             | 0.625        |
| *Phylloscopus proregulus*  | 100             | 55           | Beak length / Head-beak length / Wing length / Tail length / Body mass | 0.600        |

> **Note:** The variation in selected features reflects species-specific morphological responses to infection, as identified through feature importance analysis and cross-validation. This species-specific approach significantly improves prediction accuracy compared to using a universal feature set.

**Outputs are saved in timestamped folders, e.g., `models_multiclass_20241215_143022/`**

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

We leverage **XGBoost's built-in feature importance** (based on gain/split metrics) to identify which morphological traits contribute most to infection predictions.

---

## 🗂️ Data Availability

**Note on Data Access:**

Due to the proprietary nature of the long-term field dataset (2009–2024), the raw biological data used in this study is **not publicly available** in this repository.

* **`example_data.xlsx`**: A synthetic dataset provided for demonstration purposes, allowing users to test the pipeline structure.
* **Pre-trained Models**: We provide the full `models/` directory containing trained XGBoost models, scalers, and label encoders. You can use these to predict infection status in *your own* data without needing our raw training data.
* **Access to Raw Data**: Researchers interested in the original dataset may contact the corresponding author upon reasonable request.

---

## 🔬 Methodological Details

### Three-Round Training Protocol

To ensure optimal predictive performance, we employed an iterative feature selection strategy:

1. **Round 1:** Training with all five morphological traits
2. **Round 2:** Training with top three features based on importance scores
3. **Round 3:** Testing alternative feature combinations based on cross-species importance analysis

The model achieving highest accuracy across all rounds was selected as the final predictive model for each species.

### Temporal Validation Strategy

* **Training data:** 2009–2021 (70% training, 30% internal validation)
* **Independent test set:** 2023–2024 (temporally separated to assess real-world applicability)
* This approach prevents data leakage and provides robust evaluation for ongoing monitoring programs

---

## 📝 Citation

If you use AMHIPT in your research, please cite our paper:

> **Ruitian Xu**, **Qingfeng Gan**, **Shiqiong Chuan**, **Xi Huang***, 2026. (In Revision). *A Novel Machine Learning Model for the Prediction of Avian Haemosporidian Infection from Morphological Data*. Avian Research.

---

## 📧 Contact

For questions about the methodology or data access requests, please contact:

* **Corresponding Author:** Xi Huang ([huangxi@bnu.edu.cn](mailto:huangxi@bnu.edu.cn))
* **Lead Developer:** Ruitian Xu (GitHub: @RuitianXu; or [ruitian_xu@163.com](mailto:ruitian_xu@163.com))

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
