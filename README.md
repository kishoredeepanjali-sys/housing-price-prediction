# **🏠 California Housing Price Predictor** 

An end-to-end Machine Learning project that predicts median house values in California using a Random Forest model, with a simple interactive web app built using Streamlit.

**📖 About**

This is a beginner-friendly ML project built to understand and practice the full workflow of a machine learning project — from raw data to a working app.
The app lets you enter details like location, income, and house features, and it predicts the estimated median house value based on a trained Random Forest model.

**🧠 What I Learned / Used**

Cleaning and preparing data (handling missing values, encoding categories)
Using a scikit-learn Pipeline to keep preprocessing consistent
Training a Random Forest Regressor model
Saving the trained model using joblib
Building a simple interactive UI using Streamlit


**⚙️ Tech Stack**

Python
pandas
scikit-learn
Streamlit
joblib


# **🚀 How to Run This Project**

bash

# 1. Clone the repository
git clone https://github.com/kishoredeepanjali-sys/housing-price-prediction
cd housing-price-prediction

# 2. Install the required libraries
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
Make sure model.pkl and pipeline.pkl are in the same folder as app.py.

📂 Project Structure

├── app.py                               # Streamlit app

├── model.pkl                            # Trained Random Forest model

├── pipeline.pkl                         # Preprocessing pipeline

├── requirements.txt                     # Project dependencies

└── README.md                            # Project info

# **🖥️ What the App Does**

**You can enter:**

Location (latitude, longitude, ocean proximity)
House details (rooms, bedrooms, population, households)
Median income of the area

...and the app predicts the estimated median house value for that area.

**🔮 What I'd Like to Add Next**

 Try comparing this model with Linear Regression / XGBoost
 Add a map to pick location visually
 Improve UI design
 Deploy the app online (Streamlit Cloud)


# **🙋 About Me**

Built by **Deepanjali Kishore** as part of my Machine Learning learning journey.

📧 kishoredeepanjali@gmail.com 

🔗 [LinkedIn] 

www.linkedin.com/in/deepanjali-kishore-543a3641b

💻 [GitHub] 

https://github.com/kishoredeepanjali-sys

