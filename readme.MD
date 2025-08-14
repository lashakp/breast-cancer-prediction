# 🩺 Breast Cancer Recurrence Prediction App

A machine learning web app that predicts the **likelihood of breast cancer recurrence** using patient data.  
Built with **Python**, **Scikit-learn**, **SMOTE**, and **Gradio** for an easy-to-use interactive interface.

---

## 🚀 Features
- **Two prediction models**:
  - Random Forest Classifier
  - Logistic Regression
- **Automatic threshold tuning** for best F1 score
- **Handles class imbalance** using SMOTE
- **Feature importance visualization** (Random Forest only)
- **ROC & Precision-Recall curves** for model performance
- **Dropdown inputs** for categorical features
- **Numeric inputs** for continuous variables
- **Confidence score** for predictions

---

## 📂 Project Structure

├── breast_cancer_gradio_app_merged.py # Main script (trains or loads models, launches app)
├── breast-cancer-real.csv # Dataset
├── rf_model.pkl # Saved Random Forest model (auto-created)
├── log_model.pkl # Saved Logistic Regression model (auto-created)
├── feature_importance.png # Auto-generated chart for RF
├── roc_curve.png # ROC curve chart
├── pr_curve.png # Precision-Recall curve chart
└── README.md # Project documentation


---

## ⚙️ Installation & Setup

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

How It Works

Data preprocessing

OneHotEncoder for categorical variables

Pass-through for numerical variables

SMOTE oversampling to balance the dataset

Model training

RandomForestClassifier and LogisticRegression are trained

Best decision thresholds found for maximum F1 score

Models saved as .pkl for future fast loading

Prediction

User inputs patient details through the UI

App predicts "Recurrence" or "No Recurrence"

Confidence score is shown

For RF, top 15 important features are displayed

Visualization

ROC Curve

Precision-Recall Curve

📌 Example Use Case

This app can help doctors, researchers, or data analysts:

Explore factors influencing breast cancer recurrence

Demonstrate ML model performance visually

Test model predictions with custom patient data

⚠️ Disclaimer

This app is for educational and research purposes only.
It is not intended for medical diagnosis or treatment decisions.

📧 Contact

Created by Paul Akporarhe
GitHub: lashakp