# =========================================
# Breast Cancer Recurrence Prediction App
# Merged version: Loads pre-trained models if available,
# otherwise trains fresh and saves them for future use.
# Includes Gradio UI for predictions + visual insights.
# =========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import joblib

# Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    precision_recall_curve, roc_curve, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# =========================================
# 1️⃣ Load Dataset
# =========================================
df = pd.read_csv("breast-cancer-real.csv")  # Load CSV file into DataFrame
df.columns = [c.strip() for c in df.columns]  # Remove extra spaces in column names

# Target variable: Convert "recurrence-events" to 1, else 0
y = (df["target"].astype(str).str.strip() == "recurrence-events").astype(int)

# Features: Drop target column
X = df.drop(columns=["target"])

# Identify numeric and categorical columns
num_cols = [c for c in X.columns if np.issubdtype(df[c].dtype, np.number)]
cat_cols = [c for c in X.columns if c not in num_cols]

# Prepare dropdown choices for categorical variables
dropdown_options = {col: sorted(df[col].dropna().unique().tolist()) for col in cat_cols}

# =========================================
# 2️⃣ Preprocessing
# =========================================
# Define transformations: OneHotEncoder for categorical, passthrough for numeric
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols),
])

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# =========================================
# 3️⃣ Train Models (only if not already saved)
# =========================================
def train_and_save_models():
    print("🔄 Training models...")

    # RandomForest pipeline: preprocess → SMOTE → RF model
    rf_pipe = ImbPipeline(steps=[
        ("prep", preprocess),
        ("smote", SMOTE(random_state=42)),
        ("rf", RandomForestClassifier(
            n_estimators=600,          # Number of trees
            max_depth=8,              # Limit tree depth (prevents overfitting)
            min_samples_split=5,      # Min samples to split a node
            min_samples_leaf=2,       # Min samples in leaf node
            max_features="sqrt",      # Best practice for classification
            bootstrap=True,
            random_state=42,
            n_jobs=-1                 # Use all CPU cores
        ))
    ])
    rf_pipe.fit(X_train, y_train)

    # Logistic Regression pipeline: preprocess → SMOTE → scale → logistic regression
    log_pipe = ImbPipeline(steps=[
        ("prep", preprocess),
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler(with_mean=False)),
        ("logreg", LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",  # Helps with imbalance
            max_iter=300,
            random_state=42
        ))
    ])
    log_pipe.fit(X_train, y_train)

    # Save both models to disk
    joblib.dump(rf_pipe, "rf_model.pkl")
    joblib.dump(log_pipe, "log_model.pkl")

    print("✅ Models trained and saved.")
    return rf_pipe, log_pipe

# Check if model files exist; if not, train and save
if os.path.exists("rf_model.pkl") and os.path.exists("log_model.pkl"):
    print("📂 Loading pre-trained models...")
    rf_model = joblib.load("rf_model.pkl")
    log_model = joblib.load("log_model.pkl")
else:
    rf_model, log_model = train_and_save_models()

# =========================================
# 4️⃣ Function to Find Best F1 Threshold
# =========================================
def best_f1_threshold(pipe):
    proba = pipe.predict_proba(X_test)[:, 1]
    prec, rec, thr = precision_recall_curve(y_test, proba)
    # Compute F1 for each threshold
    f1s = np.where((prec + rec) > 0, 2 * (prec * rec) / (prec + rec), 0)
    best_idx = np.argmax(f1s)
    return thr[best_idx] if best_idx < len(thr) else 0.5

rf_best_thr = round(best_f1_threshold(rf_model), 2)
log_best_thr = round(best_f1_threshold(log_model), 2)

# =========================================
# 5️⃣ Prediction Function
# =========================================
def predict_case(model_name, threshold, *features):
    # Create DataFrame for single patient input
    input_dict = {}
    for col, val in zip(cat_cols + num_cols, features):
        input_dict[col] = [val]
    input_df = pd.DataFrame(input_dict)

    # Choose model
    model = rf_model if model_name == "RandomForest" else log_model

    # Predict probability of recurrence
    proba = model.predict_proba(input_df)[:, 1][0]
    prediction = 1 if proba >= threshold else 0

    label = "Recurrence" if prediction == 1 else "No Recurrence"
    confidence = round(proba * 100, 1)

    # Feature importance for RF model
    fig_path = None
    if model_name == "RandomForest":
        ohe = model.named_steps["prep"].named_transformers_["cat"]
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        all_feature_names = np.concatenate([cat_feature_names, num_cols])
        importances = model.named_steps["rf"].feature_importances_
        feat_imp = pd.DataFrame({"feature": all_feature_names, "importance": importances})
        feat_imp = feat_imp.sort_values("importance", ascending=False).head(15)

        plt.figure(figsize=(6, 4))
        plt.barh(feat_imp["feature"], feat_imp["importance"])
        plt.gca().invert_yaxis()
        plt.title("Top 15 Feature Importances (RF)")
        plt.tight_layout()
        fig_path = "feature_importance.png"
        plt.savefig(fig_path)
        plt.close()

    return label, f"{confidence}%", fig_path

# =========================================
# 6️⃣ Generate ROC & PR Curve Visualizations
# =========================================
def generate_visuals():
    # ROC Curve
    plt.figure(figsize=(6, 5))
    for name, model in [("RF", rf_model), ("Logistic", log_model)]:
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()

    # Precision-Recall Curve
    plt.figure(figsize=(6, 5))
    for name, model in [("RF", rf_model), ("Logistic", log_model)]:
        proba = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, proba)
        plt.plot(rec, prec, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pr_curve.png")
    plt.close()

# Call once to create static images
generate_visuals()

# =========================================
# 7️⃣ Build Gradio Interface
# =========================================
with gr.Blocks() as app:
    gr.Markdown("# 🩺 Breast Cancer Recurrence Prediction")

    with gr.Tabs():
        # ---- Prediction Tab ----
        with gr.TabItem("Predict"):
            model_choice = gr.Dropdown(
                ["RandomForest", "Logistic Regression"],
                value="RandomForest", label="Select Model"
            )
            threshold_slider = gr.Slider(
                0.05, 0.6, value=rf_best_thr, step=0.01, label="Decision Threshold"
            )

            # Create input fields for all features
            inputs = []
            for col in cat_cols:
                inputs.append(gr.Dropdown(choices=dropdown_options[col], label=col))
            for col in num_cols:
                inputs.append(gr.Number(label=col))

            predict_btn = gr.Button("Predict")
            output_label = gr.Textbox(label="Prediction")
            output_conf = gr.Textbox(label="Confidence")
            output_plot = gr.Image(label="Feature Importance (RF only)")

            predict_btn.click(
                fn=predict_case,
                inputs=[model_choice, threshold_slider] + inputs,
                outputs=[output_label, output_conf, output_plot]
            )

        # ---- Visual Insights Tab ----
        with gr.TabItem("Visual Insights"):
            gr.Image("roc_curve.png", label="ROC Curve")
            gr.Image("pr_curve.png", label="Precision-Recall Curve")

        # ---- About Tab ----
        with gr.TabItem("About Model"):
            gr.Markdown(f"""
            **How it works:**
            - Automatically loads saved models if available.
            - If not, trains new ones and saves them.
            - Data balanced with SMOTE, categorical features one-hot encoded.
            - Best F1 thresholds:
              - RF: **{rf_best_thr}**
              - Logistic: **{log_best_thr}**
            """)

# Launch Gradio app
app.launch()