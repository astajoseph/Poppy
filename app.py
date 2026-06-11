

import streamlit as st
import pandas as pd
import sqlite3
import joblib
import numpy as np
import os


DB_PATH = "E:/poppy/OP.db"
MODEL_PATH = "E:/poppy/model.h5"
SCALER_PATH = "E:/poppy/scaler.pkl"
IMPUTER_PATH = "E:/poppy/imputer.pkl"

def load_artifacts():
    model = None
    scaler = None
    imputer = None
    if os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH, compile=False)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(IMPUTER_PATH):
        imputer = joblib.load(IMPUTER_PATH)
    return model, scaler, imputer


def load_db_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM training_data", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

def get_feature_columns():
    
    df = pd.read_csv("E:/poppy/Final_Training_Dataset.csv", nrows=1)
    drop_cols = ["Kg_per_Hectare", "Yield_std", "Yield_min", "Yield_max", "Village", "District", "State"]
    return [col for col in df.columns if col not in drop_cols]

def export_db_data(df):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "crop_data.csv", "text/csv")


def run_gee_extraction():
    st.info("GEE extraction is a stub. Integrate your logic here.")

def predict_yield(input_dict, model, scaler, imputer):
    try:
        X = pd.DataFrame([input_dict])
        X_imp = imputer.transform(X)
        X_scaled = scaler.transform(X_imp)
        pred = model.predict(X_scaled)
        return float(pred.flatten()[0])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# --- Streamlit UI ---
st.set_page_config(page_title="Opium Poppy MQY Prediction", layout="wide")

# --- Sidebar ---
st.sidebar.image(r"E:\poppy\opium.jpeg", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Database Table", "Yield Prediction", "About"])
st.sidebar.markdown("---")
st.sidebar.info("Developed for Opium Poppy MQY Prediction. Use the tabs to navigate between data and prediction.")

# --- Main Header ---
st.markdown("""
    <h1 style='text-align: center; color: #2E4053; margin-bottom: 1.5rem;'>Opium Poppy MQY Prediction</h1>
    <p style='text-align: center; color: #555; font-size:1.1rem;'>A modern web app for crop yield prediction and data exploration.</p>
""", unsafe_allow_html=True)

# --- Load model artifacts and data ---
model, scaler, imputer = load_artifacts()
db_df = load_db_data()
feature_names = get_feature_columns()
selected_row = None

if page == "Database Table":
    st.subheader("Database Table: crop_data")
    if not db_df.empty:
        st.dataframe(db_df, use_container_width=True, height=400)
        export_db_data(db_df)
        row_idx = st.number_input("Select row index for autofill", min_value=0, max_value=len(db_df)-1, step=1, value=0)
        selected_row = db_df.iloc[row_idx]
        st.info("Switch to the 'Yield Prediction' tab to autofill prediction inputs from this row.")
    else:
        st.warning("No data found or database error.")

elif page == "Yield Prediction":
    st.subheader("Yield Prediction")
    if model is None or scaler is None or imputer is None:
        st.error("Model or preprocessing artifacts not found. Please train and save them in E:/poppy/.")
    else:
        autofill = st.checkbox("Autofill from database row", value=False)
        input_dict = {}
        if autofill and db_df is not None and not db_df.empty:
            row_idx = st.number_input("Select row index for autofill", min_value=0, max_value=len(db_df)-1, step=1, value=0)
            selected_row = db_df.iloc[row_idx]
            for feat in feature_names:
                default_val = float(selected_row[feat]) if feat in selected_row else 0.0
                input_dict[feat] = st.number_input(feat, value=default_val)
        else:
            cols = st.columns(2)
            for i, feat in enumerate(feature_names):
                with cols[i % 2]:
                    input_dict[feat] = st.number_input(feat, value=0.0)
        if st.button("Predict Yield"):
            pred = predict_yield(input_dict, model, scaler, imputer)
            if pred is not None:
                st.success(f"🌱 Predicted Yield: {pred:.2f} Kg/Ha")

elif page == "About":
    st.subheader("About This App")
    st.markdown("""
        <p style='font-size:1.1rem;'>
        This web application predicts the Minimum Qualifying Yield (MQY) for opium poppy crops using a machine learning model. You can explore the training data, autofill prediction inputs from the database, and get instant yield predictions.<br><br>
        <b>Developed by:</b> Your Team Name<br>
        <b>Contact:</b> your@email.com
        </p>
    """, unsafe_allow_html=True)
