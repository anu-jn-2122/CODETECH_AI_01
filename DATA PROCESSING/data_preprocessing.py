# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Example raw data (for demonstration)
data = {
    'Age': [25, np.nan, 35, 45, 29],
    'Salary': [50000, 60000, np.nan, 80000, 62000],
    'Gender': ['Male', 'Female', 'Female', np.nan, 'Male'],
    'Purchased': ['No', 'Yes', 'No', 'Yes', 'Yes']
}

# Step 1: Create a DataFrame
df = pd.DataFrame(data)

# Display the raw data
print("Raw Data:")
print(df)

# Step 2: Handle Missing Data (Imputation)
# We'll fill missing numerical values with the mean, and missing categorical values with the most frequent category.
numeric_features = ['Age', 'Salary']
categorical_features = ['Gender']

# Preprocessing pipeline for numerical features (fill missing values with mean, and standardize)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing pipeline for categorical features (fill missing values and one-hot encode)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps for both numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 3: Create the Pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Step 4: Split the Data into Features (X) and Target (y)
X = df.drop('Purchased', axis=1)  # Features (everything except 'Purchased')
y = df['Purchased']  # Target variable (the label)

# Step 5: Split the Data into Training and Test Sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Apply Preprocessing Pipeline to the Training Data
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Display the processed training and test data
print("\nProcessed Training Data:")
print(X_train_processed)

print("\nProcessed Test Data:")
print(X_test_processed)

# Display the target variable (y) for training and testing
print("\nTarget for Training:")
print(y_train)

print("\nTarget for Testing:")
print(y_test)
