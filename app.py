
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Kidney Disease AI",
    page_icon="🩺"
)

st.title("🩺 Kidney Disease Risk Classifier")
st.write("Upload a CSV file containing the model's input features.")

st.warning(
    "Educational demo only. This model is not validated "
    "for medical diagnosis or treatment."
)

MODEL_PATH = Path(__file__).parent / "kidney_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if "Target" in data.columns:
            data = data.drop(columns=["Target"])

        if not hasattr(model, "feature_names_in_"):
            st.error(
                "Model ke expected feature names available nahi hain. "
                "Model ke mutabiq app code adjust karna hoga."
            )
            st.stop()

        expected = list(model.feature_names_in_)
        missing = [col for col in expected if col not in data.columns]
        extra = [col for col in data.columns if col not in expected]

        if missing:
            st.error("CSV mein yeh columns missing hain:")
            st.write(missing)
        elif extra:
            st.error("CSV mein extra columns hain:")
            st.write(extra)
        else:
            data = data[expected]
            st.dataframe(data.head())

            if st.button("Predict Risk Class"):
                predictions = model.predict(data)
                result = pd.DataFrame({
                    "Predicted_Class": predictions
                })

                st.subheader("Prediction Results")
                st.dataframe(result)

                st.download_button(
                    "Download Results",
                    result.to_csv(index=False).encode("utf-8"),
                    "kidney_predictions.csv",
                    "text/csv"
                )

    except Exception as e:
        st.error(f"CSV error: {e}")
