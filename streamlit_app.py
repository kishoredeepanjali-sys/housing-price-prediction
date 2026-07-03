import streamlit as st
import pandas as pd
import joblib
import os

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

st.set_page_config(page_title="Housing Price Prediction", page_icon=":house:", layout="wide")   
st.title("Housing Price Prediction App")
st.write("This app predicts the median house value based on various features of the housing dataset.")

# Load the model and pipeline
@st.cache_resource
def load_model_and_pipeline():
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    return model, pipeline
try:
    model, pipeline = load_model_and_pipeline()
    model_loaded = True
except FileNotFoundError:
    st.error("Model or pipeline file not found.")
    model_loaded = False
if model_loaded:
    with st.form("prediction_form"):
        st.subheader("Input Features")

        col1, col2 = st.columns(2)
        with col1:
            longitude = st.number_input("Longitude", value=-122.23,format="%.4f")
            latitude = st.number_input("Latitude", value=37.88,format="%.4f")
            housing_median_age = st.number_input("Housing Median Age", value=41.0,min_value=0.0,format="%.1f")
            total_rooms = st.number_input("Total Rooms", value=880.0,min_value=0.0,format="%.1f")
            total_bedrooms = st.number_input("Total Bedrooms", value=129.0,min_value=0.0,format="%.1f")
        with col2:
            population = st.number_input("Population", value=322.0,min_value=0.0,format="%.1f")
            households = st.number_input("Households", value=126.0,min_value=0.0,format="%.1f")
            median_income = st.number_input("Median Income", value=8.3252,min_value=0.0,format="%.4f")
            ocean_proximity = st.selectbox("Ocean Proximity", options=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

            submitted = st.form_submit_button("Predict Price 🔮")

        if submitted:
            input_data = pd.DataFrame({
                'longitude': [longitude],
                'latitude': [latitude],
                'housing_median_age': [housing_median_age],
                'total_rooms': [total_rooms],
                'total_bedrooms': [total_bedrooms],
                'population': [population],
                'households': [households],
                'median_income': [median_income],
                'ocean_proximity': [ocean_proximity]
            })

            transformed_input = pipeline.transform(input_data)
            prediction = model.predict(transformed_input)[0]

            st.success(f"Predicted Median House Value: ${prediction:,.2f}")