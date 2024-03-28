#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

# Load the Excel file
df = pd.read_excel('AnomaData.xlsx', None)

# Print the names of the sheets to identify the correct one for time series data
sheet_names = df.keys()
print('Sheet names:', sheet_names)

time_series_data = df['Sheet1']

# Displaying the head of the time series data
print(time_series_data.head())


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the time series data
data = pd.read_excel('AnomaData.xlsx')

# Display the first few rows of the data
print(data.head())


# In[5]:


# Performing exploratory data analysis on the time series data

# Summary statistics
summary_stats = data.describe()

# Check for missing values
missing_values = data.isnull().sum()

# Display summary statistics and missing values
print('Summary Statistics:\
', summary_stats)
print('\
Missing Values:\
', missing_values)



# In[33]:


# Check the data quality and identify any issues

# Check for duplicate rows
duplicate_rows = data.duplicated().sum()


# In[34]:


# Check for constant columns
constant_columns = data.columns[data.nunique() <= 1]



# In[35]:


# Treat missing values in the data

# Fill missing values with the mean of each column
data_filled = data.fillna(data.mean())

# Display the first few rows of the filled data
print(data_filled.head())


# In[36]:


# First, let's identify the columns that are causing the issue by checking the data types of each column in `data_filled`.
print(data_filled.dtypes)


# In[37]:


# After identifying the datetime columns, we will exclude them from the Z-score calculation.
# This code will be adjusted based on the output of `data_filled.dtypes`.


# Convert the 'time' column to datetime format
import pandas as pd
from tqdm import tqdm

data_filled['time'] = pd.to_datetime(data_filled['time'])
print('Conversion to datetime complete.')


# In[38]:


from scipy.stats import zscore

# Exclude the 'time' column from Z-score calculation as it's not numeric
numeric_cols = data_filled.select_dtypes(include=['float64', 'int64']).columns

# Calculate Z-scores for the numeric columns
data_filled[numeric_cols] = data_filled[numeric_cols].apply(zscore)

print('Z-scores calculated for numeric columns.')


# In[39]:


# Feature Engineering
# For this dataset, let's create a new feature that could potentially be useful.
# We will create a feature that represents the mean of all numeric features for each row.

# Selecting numeric columns (excluding 'time' as it's not numeric)
numeric_cols = data_filled.select_dtypes(include=['float64', 'int64']).columns

# Calculating the mean of numeric features for each row
data_filled['mean_numeric'] = data_filled[numeric_cols].mean(axis=1)

print('Feature engineering complete. New feature "mean_numeric" added.')



# In[40]:


from sklearn.feature_selection import SelectKBest, f_classif

# Assuming 'y' is the target variable
X = data_filled.drop(['time', 'y'], axis=1)  # Dropping 'time' as it's not a feature and 'y' as it's the target
y = data_filled['y']


# In[41]:


# Select top 10 features based on ANOVA F-value
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)

# Get the selected feature names
selected_features = X.columns[selector.get_support()]

print('Selected features:', selected_features.tolist())


# In[43]:


#The conversion of the 'time' column to datetime format has been successfully completed, and Z-scores for numeric columns have been calculated. Following feature engineering, a new feature named "mean_numeric" was added, representing the mean of all numeric features for each row.
#For feature selection, the top 10 features based on ANOVA F-value were selected. These features are:
#['x2', 'x3', 'x15', 'x17', 'x18', 'x19', 'x52', 'x54', 'y.1', 'mean_numeric']
#These selected features are considered the most relevant for predicting the target variable 'y' based on their statistical significance.


# In[44]:


from sklearn.model_selection import train_test_split

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Data split into training and testing sets.')


# In[45]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Metrics for model evaluation
# Since we're dealing with a classification problem, we'll use the following metrics:
# - Accuracy: Overall how often the model predicts correctly
# - Precision: When it predicts the positive class, how often is it correct?
# - Recall: Out of all the positive classes, how many did the model predict correctly?
# - F1 Score: A weighted average of Precision and Recall

print('Selected metrics for model evaluation: Accuracy, Precision, Recall, F1 Score.')


# In[46]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Model selection using Random Forest Classifier
rf = RandomForestClassifier()


# In[47]:


# Hyperparameter tuning using Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30]
}

grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print('Model selection and hyperparameter tuning complete.')


# In[48]:


# Inspecting the target variable 'y'
print(y_train.describe())
print('\
Unique values in y:', len(np.unique(y_train)))



# In[ ]:


#Given the nature of the target variable 'y', which appears to be continuous, we should proceed with a regression model instead of a classifier. Let's select and tune a Random Forest Regressor model for this task


from sklearn.ensemble import RandomForestRegressor

# Model selection using Random Forest Regressor
rf_regressor = RandomForestRegressor()
# Hyperparameter tuning using Grid Search
param_grid_regressor = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30]
}

grid_search_regressor = GridSearchCV(rf_regressor, param_grid_regressor, cv=5)
grid_search_regressor.fit(X_train, y_train)

print('Model selection and hyperparameter tuning for regression complete.')



# In[ ]:


from sklearn.model_selection import GridSearchCV

# Model selection using Random Forest Regressor
rf_regressor = RandomForestRegressor()

# Hyperparameter tuning using Grid Search
param_grid_regressor = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30]
}

grid_search_regressor = GridSearchCV(rf_regressor, param_grid_regressor, cv=5)
grid_search_regressor.fit(X_train, y_train)

print('Model selection and hyperparameter tuning for regression complete.')


# In[ ]:




