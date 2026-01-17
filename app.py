import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Credit Risk Analyzer", layout="wide")

st.title("📊 Credit Default Prediction System")
st.markdown("""
This application allows you to upload client data and predict the probability of credit default 
using various Machine Learning models.
""")

# --- SIDEBAR: MODEL SELECTION ---
st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox(
    "Choose a Classifier:",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# --- HELPER: LOAD ASSETS ---
@st.cache_resource
def load_prediction_tools(model_name):
    """Loads the saved model and the scaler."""
    file_path = f'model/{model_name.replace(" ", "_").lower()}.pkl'
    try:
        loaded_model = pickle.load(open(file_path, 'rb'))
        loaded_scaler = pickle.load(open('model/scaler_custom.pkl', 'rb'))
        return loaded_model, loaded_scaler
    except FileNotFoundError:
        st.error("Model files not found. Ensure 'model/' folder is uploaded.")
        return None, None

# --- MAIN APP FUNCTIONALITY ---
# [cite: 91] Dataset upload option
upload_file = st.file_uploader("Upload CSV File (Test Data)", type=["csv"])

if upload_file is not None:
    try:
        # Load User Data
        data = pd.read_csv(upload_file)
        
        # Display Data Preview
        st.subheader("Data Preview")
        st.dataframe(data.head())

        # Preprocessing: Handle ID column if present
        if 'ID' in data.columns:
            data_processed = data.drop('ID', axis=1)
        else:
            data_processed = data.copy()

        # Identify Target Column (if it exists)
        target_col = 'default payment next month'
        if target_col not in data_processed.columns and 'IsDefaulter' in data_processed.columns:
             target_col = 'IsDefaulter'
        
        # Separate Features
        if target_col in data_processed.columns:
            X_input = data_processed.drop(target_col, axis=1)
            y_true = data_processed[target_col]
            has_ground_truth = True
        else:
            X_input = data_processed
            has_ground_truth = False

        # Load Model
        model, scaler = load_prediction_tools(selected_model)

        if model:
            # Scale Data (Must use the same scaler as training!)
            X_scaled = scaler.transform(X_input)
            
            # Make Predictions
            preds = model.predict(X_scaled)
            
            # Add predictions to dataframe
            results_df = data.copy()
            results_df['Predicted_Default'] = preds
            
            st.subheader(f"Predictions using {selected_model}")
            st.write(results_df.head())
            
            # --- EVALUATION METRICS [cite: 93, 94] ---
            if has_ground_truth:
                st.divider()
                st.header("Model Performance Evaluation")
                
                # 1. Metrics Display
                acc = accuracy_score(y_true, preds)
                st.metric(label="Accuracy Score", value=f"{acc:.4f}")
                
                col1, col2 = st.columns(2)
                
                # 2. Confusion Matrix
                with col1:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_true, preds)
                    fig_cm, ax_cm = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax_cm)
                    st.pyplot(fig_cm)
                
                # 3. Classification Report
                with col2:
                    st.subheader("Classification Report")
                    report = classification_report(y_true, preds, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())
            else:
                st.info("Upload a dataset with labels ('default payment next month') to see evaluation metrics.")
                
    except Exception as e:
        st.error(f"An error occurred: {e}")