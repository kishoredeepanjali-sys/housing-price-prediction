import pandas as pd 
import numpy as np       
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# 1 loading the dataset
housing = pd.read_csv('housing.csv')

# 2 create a stratified testset
housing['income_cat'] = pd.cut(housing['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop('income_cat', axis=1) # we will use this for training and work on it
    strat_test_set = housing.loc[test_index].drop('income_cat', axis=1) # set aside the test set for later use

# we will work on the copy of the training set to avoid messing up the original data
housing = strat_train_set.copy()


# 3 list the predictors/features and the labels
housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value', axis=1)

print(housing,housing_labels)


# 4 separate the numerical and categorical columns
num_attribs = housing.drop('ocean_proximity', axis=1).columns.to_list()
cat_attribs = ['ocean_proximity']

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

# 6 let's transform the data using the full pipeline
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
print(housing_prepared.shape)

# 7 let's train a model using the training set
# we will use a linear regression model

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_predictions = lin_reg.predict(housing_prepared)
#lin_rmse = root_mean_squared_error(housing_labels, lin_predictions)
lin_rmses = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
#print("Linear Regression RMSE:", lin_rmse)
print(pd.Series(lin_rmses).describe())

# decision tree model
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
tree_predictions = tree_reg.predict(housing_prepared)
#tree_rmse = root_mean_squared_error(housing_labels, tree_predictions)
tree_rmses = -cross_val_score(tree_reg, housing_prepared, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
#print("Decision Tree RMSE:", tree_rmses.mean())
print(pd.Series(tree_rmses).describe())
# random forest model
rf_reg = RandomForestRegressor()
rf_reg.fit(housing_prepared, housing_labels)
rf_predictions = rf_reg.predict(housing_prepared)
#rf_rmse = root_mean_squared_error(housing_labels, rf_predictions)
rf_rmses = -cross_val_score(rf_reg, housing_prepared, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
#print("Random Forest RMSE:", rf_rmse)
print(pd.Series(rf_rmses).describe())

