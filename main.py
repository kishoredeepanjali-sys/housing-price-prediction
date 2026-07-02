import os
import joblib
import pandas as pd
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    # 5 let's create a pipeline for
    #  the numerical attributes
    num_pipeline = Pipeline([('simple_imputer', SimpleImputer(strategy='median')),
                             ('std_scaler', StandardScaler())])

    # the categorical attributes
    cat_pipeline = Pipeline([('simple_imputer', SimpleImputer(strategy='most_frequent')),   
                             ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))])

    # construct a full pipeline for both numerical and categorical attributes
    full_pipeline = ColumnTransformer([('num', num_pipeline, num_attribs),
                                       ('cat', cat_pipeline, cat_attribs)])
    
    return full_pipeline    
if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
#lets train the model and save it to disk
    # 1 loading the dataset
    housing = pd.read_csv('housing.csv')

    # 2 create a stratified testset
    housing['income_cat'] = pd.cut(housing['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing.loc[test_index].drop('income_cat', axis=1).to_csv('input.csv', index=False) # we will use this for training and work on it
        housing = housing.loc[train_index].drop('income_cat', axis=1) # set aside the test set for later use
    

    # 3 list the predictors/features and the labels
    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value', axis=1)

    #print(housing_features,housing_labels)


    # 4 separate the numerical and categorical columns
    num_attribs = housing_features.drop('ocean_proximity', axis=1).columns.to_list()
    cat_attribs = ['ocean_proximity']

    full_pipeline = build_pipeline(num_attribs, cat_attribs)

    # 6 let's transform the data using the full pipeline
    housing_prepared = full_pipeline.fit_transform(housing)

    #print(housing_prepared)
    #print(housing_prepared.shape)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE, compress=3)
    joblib.dump(full_pipeline, PIPELINE_FILE, compress=3)
    print(f"Model and pipeline saved to {MODEL_FILE} and {PIPELINE_FILE}") 
    print("Model training completed.")
else:
    print(f"Model and pipeline already exist. Loading from {MODEL_FILE} and {PIPELINE_FILE}")
    model = joblib.load(MODEL_FILE)
    full_pipeline = joblib.load(PIPELINE_FILE)
    input_data = pd.read_csv('input.csv')
    transformed_input = full_pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions
    input_data.to_csv('output.csv', index=False)
    print("Inference completed. Predictions saved to output.csv")


    print(f"Predictions for input data: {predictions}")
