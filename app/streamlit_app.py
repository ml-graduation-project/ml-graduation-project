import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys
from io import StringIO
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessor import DataPreprocessor
from src.feature_engineer import FeatureEngineer


st.set_page_config(page_title="RUL Prediction App", layout="wide")

st.title("🛠️ Remaining Useful Life (RUL) Prediction App")
st.write("Upload your test data (`.txt` or `.csv`) to get RUL predictions using the trained LSTM model.")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_lstm_fd001.keras")
SCALER_PATH = os.path.join(BASE_DIR, "artifacts", "scaler.pkl")

try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error(f"Could not load model or scaler. Please check your paths.\n\n{e}")
    st.stop()
    
index = ["engine", "cycle"]
setting = [f"op_setting_{i}" for i in range(1, 4)]
sensor = [
    "Fan inlet temperature (◦R)",
    "LPC outlet temperature (◦R)",
    "HPC outlet temperature (◦R)",
    "LPT outlet temperature (◦R)",
    "Fan inlet Pressure (psia)",
    "Bypass-duct pressure (psia)",
    "HPC outlet pressure (psia)",
    "Physical fan speed (rpm)",
    "Physical core speed (rpm)",
    "Engine pressure ratio (P50/P2)",
    "HPC outlet Static pressure (psia)",
    "Ratio of fuel flow to Ps30 (pps/psia)",
    "Corrected fan speed (rpm)",
    "Corrected core speed (rpm)",
    "Bypass Ratio",
    "Burner fuel-air ratio",
    "Bleed Enthalpy",
    "Required fan speed",
    "Required fan conversion speed",
    "High-pressure turbines Cool air flow",
    "Low-pressure turbines Cool air flow",
]
col_names = index + setting + sensor


def create_test_last_sequences(df, seq_length, features):
    X, ids = [], []
    for eid in sorted(df["engine"].unique()):
        ed = df[df["engine"] == eid]
        if len(ed) >= seq_length:
            X.append(ed[features].iloc[-seq_length:].values)
            ids.append(eid)
    return np.array(X), ids



uploaded_file = st.file_uploader("Upload your test data file", type=["txt", "csv"])

if uploaded_file is not None:
    try:
     
        if uploaded_file.name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, sep=r"\s+", header=None, names=col_names)
        else:
            df = pd.read_csv(uploaded_file)

        st.success("File successfully loaded!")
        st.write("### Preview of uploaded data:")
        st.dataframe(df.head())

        useful_sensors = scaler.feature_names_in_
        df_features = df[useful_sensors].copy()

        
        df_features_scaled = pd.DataFrame(
            scaler.transform(df_features),
            columns=useful_sensors
        )

        
        df_scaled = pd.concat([df[["engine", "cycle"]].reset_index(drop=True), df_features_scaled], axis=1)

        st.info("Features successfully scaled and aligned with training data.")

        sequence_length = 50
        X_test, test_ids = create_test_last_sequences(df_scaled, sequence_length, useful_sensors)

        if len(X_test) == 0:
            st.warning("Not enough data per engine to create sequences (need at least 50 cycles).")
        else:
          
            y_pred = model.predict(X_test)
            results_df = pd.DataFrame({
                "Engine ID": test_ids,
                "Predicted RUL": y_pred.flatten()
            })

            st.success(" Prediction complete!")
            st.write("### Predicted RUL per engine:")
            st.dataframe(results_df)
            
            st.write("---")
            st.subheader(" Evaluate Model Accuracy (Optional)")
            true_rul_file = st.file_uploader("Upload the true RUL file (e.g. RUL_FD001.txt)", type=["txt", "csv"], key="true_rul")

            if true_rul_file is not None:
                try:
        
                    true_rul_df = pd.read_csv(true_rul_file, sep=r"\s+", header=None, names=["RUL"])
                    true_rul_df["Engine ID"] = np.arange(1, len(true_rul_df) + 1)

        
                    eval_df = pd.merge(results_df, true_rul_df, on="Engine ID", how="inner")
                    eval_df.rename(columns={"RUL": "True RUL"}, inplace=True)

 
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

                    rmse = np.sqrt(mean_squared_error(eval_df["True RUL"], eval_df["Predicted RUL"]))
                    mae = mean_absolute_error(eval_df["True RUL"], eval_df["Predicted RUL"])
                    r2 = r2_score(eval_df["True RUL"], eval_df["Predicted RUL"])

                    st.write("###  Evaluation Results:")
                    st.metric("RMSE", f"{rmse:.2f}")
                    st.metric("MAE", f"{mae:.2f}")
                    st.metric("R² Score", f"{r2:.3f}")

                    st.write("### Comparison Table:")
                    st.dataframe(eval_df)

                    
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.plot(eval_df["Engine ID"], eval_df["True RUL"], label="True RUL", marker="o", color="blue")
                    ax2.plot(eval_df["Engine ID"], eval_df["Predicted RUL"], label="Predicted RUL", marker="x", color="red", linestyle="--")
                    ax2.set_xlabel("Engine ID")
                    ax2.set_ylabel("RUL")
                    ax2.set_title("Predicted vs True RUL per Engine")
                    ax2.legend()
                    st.pyplot(fig2)

                except Exception as e:
                    st.error(f" Error evaluating model: {e}")
            else:
                st.info("You can optionally upload a true RUL file to evaluate model accuracy.")

            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(results_df["Engine ID"], results_df["Predicted RUL"], color="skyblue")
            ax.set_xlabel("Engine ID")
            ax.set_ylabel("Predicted RUL")
            ax.set_title("Predicted Remaining Useful Life per Engine")
            st.pyplot(fig)

           
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=" Download Predictions as CSV",
                data=csv,
                file_name="rul_predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f" Error processing file: {e}")
else:
    st.info("Please upload a `.txt` or `.csv` file to begin.")
