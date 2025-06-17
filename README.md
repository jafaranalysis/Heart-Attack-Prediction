#  Heart Attack Prediction using Machine Learning

This is a complete end-to-end machine learning project that predicts the likelihood of a heart attack based on clinical data. The project follows a structured workflow based on academic ML project guidelines.

---

##  Dataset

- **Source**: Kaggle - Heart Attack Clinical Dataset
- **Records**: 1,319 samples
- **Features**: Age, blood pressure, glucose, troponin levels, etc.
- **Target**: `class` — whether the patient has had a heart attack (0 = No, 1 = Yes)

---

##  Project Objective

To build a classification model that can predict whether a patient has suffered a heart attack using structured clinical data. This can help doctors in early diagnosis and preventive treatment planning.

---

##  Problem Framing

- **Type**: Supervised binary classification
- **Goal**: Predict `class` (0 or 1)
- **Evaluation Metric**: `F1-Score`  
  *(Chosen for its balance between precision and recall, which is important in healthcare problems)*

---

## 🏗️ ML Workflow Steps

1. **Data Loading & Exploration**
   - Loaded CSV using pandas
   - Verified shape, data types, and missing values
2. **EDA**
   - Visualized distributions and correlations
   - Checked for class imbalance and outliers
3. **Data Cleaning**
   - Converted categorical target to binary numeric (0/1)
4. **Feature Engineering**
   - Created a new `bp_diff` feature (systolic - diastolic pressure)
5. **Preprocessing**
   - Train/Validation/Test split (70/15/15)
   - Applied `StandardScaler` to numeric features
6. **Model Training**
   - Trained 5 models:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Gradient Boosting
     - SVM
7. **Model Selection & Tuning**
   - Used `GridSearchCV` with `F1-Score` as scorer
   - Tuned Random Forest (best model)
8. **Final Evaluation**
   - Evaluated on test set using `classification_report`

---

## ✅ Final Results

- **Best Model**: Random Forest (after tuning)
- **Test Accuracy**: 99%
- **Test F1-Score**: 0.99 (for both classes)

---

---

## 📦 Files Included

- `Heart_Attack_Prediction.ipynb` — Jupyter Notebook with code + analysis
- `Heart Attack.csv` — Dataset used
- `README.md` — This file

---

## 🧪 How to Run

```bash
# Required Libraries
pip install pandas numpy scikit-learn matplotlib seaborn
