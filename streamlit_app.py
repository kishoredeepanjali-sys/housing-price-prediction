import streamlit as st
import pandas as pd
import joblib

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

USD_TO_INR = 83  # approximate conversion rate, update if needed

st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 California Housing Price Predictor")
st.write("This app predicts the median house value in California based on various features. Adjust the sliders and select options to see the predicted price.")
@st.cache_resource
def load_model_and_pipeline():
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    return model, pipeline

try:
    model, pipeline = load_model_and_pipeline()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("model.pkl or pipeline.pkl not found. Please place model.pkl and pipeline.pkl in the same folder.")

if model_loaded:
    with st.form("prediction_form"):
        st.subheader("📍 Location")
        col1, col2 = st.columns(2)
        with col1:
            longitude = st.slider("Longitude", min_value=-124.5, max_value=-114.0, value=-122.23, step=0.01)
        with col2:
            latitude = st.slider("Latitude", min_value=32.5, max_value=42.0, value=37.88, step=0.01)

        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"]
        )

        st.subheader("🏘️ Property Details")
        col3, col4 = st.columns(2)
        with col3:
            housing_median_age = st.slider("Housing Median Age (years)", min_value=1, max_value=52, value=41)
            total_rooms = st.slider("Total Rooms", min_value=1, max_value=10000, value=880, step=10)
            total_bedrooms = st.slider("Total Bedrooms", min_value=1, max_value=2000, value=129, step=5)

        with col4:
            population = st.slider("Population", min_value=1, max_value=10000, value=322, step=10)
            households = st.slider("Households", min_value=1, max_value=2000, value=126, step=5)
            median_income = st.slider("Median Income (in $10,000s)", min_value=0.5, max_value=15.0, value=8.3252, step=0.01)

        submitted = st.form_submit_button("Predict Price 🔮")

    if submitted:
        input_df = pd.DataFrame([{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity,
        }])

        transformed = pipeline.transform(input_df)
        prediction_usd = model.predict(transformed)[0]
        prediction_inr = prediction_usd * USD_TO_INR

        st.success(f"### Predicted House Value: ₹{prediction_inr:,.0f}")
        st.caption(f"(Approx. conversion from ${prediction_usd:,.2f} USD at ₹{USD_TO_INR}/USD)")