# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report
import streamlit as st

# Load the dataset
url = "https://docs.google.com/spreadsheets/d/18eR6DBe5TMWU9FnIewaGtsepDbV4BOyr/export?format=csv"
df = pd.read_csv(url)

# Step 1: Data Exploration
print(df.head())
print(df.info())
print(df.describe())

# Step 2: Data Preprocessing
# Handling missing values if any
df.dropna(inplace=True)

# Step 3: Exploratory Data Analysis
# Explore skewness and outliers
sns.boxplot(x='selling_price', data=df)
plt.show()

# Step 4: Splitting the dataset into features and target variable
X = df.drop(['id', 'item_date', 'customer', 'country', 'status', 'selling_price'], axis=1)
y_regression = df['selling_price']

# Step 5: ML Regression Model
# Splitting the data into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Linear Regression Model
reg_model = LinearRegression()
reg_model.fit(X_train_reg_scaled, y_train_reg)
reg_pred = reg_model.predict(X_test_reg_scaled)
reg_mse = mean_squared_error(y_test_reg, reg_pred)
print('Regression Mean Squared Error:', reg_mse)

# Step 6: ML Classification Model
# Encoding the 'status' column
df['status_encoded'] = df['status'].map({'LOST': 0, 'WON': 1})

X_classification = df.drop(['id', 'item_date', 'customer', 'country', 'status', 'selling_price', 'status_encoded'], axis=1)
y_classification = df['status_encoded']

# Splitting the data into training and testing sets
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Random Forest Classifier
cls_model = RandomForestClassifier()
cls_model.fit(X_train_cls, y_train_cls)
cls_pred = cls_model.predict(X_test_cls)
print('Classification Report:\n', classification_report(y_test_cls, cls_pred))

# Step 7: Streamlit Integration
st.title('Industrial Copper Modeling')

# Sidebar for user input
st.sidebar.header('Input Parameters')

# Create input fields for each column
columns = ['quantity tons', 'item type', 'application', 'thickness', 'width', 'material_ref', 'product_ref']
input_data = {}
for col in columns:
    input_data[col] = st.sidebar.text_input(col, '')

# Function to make predictions
def predict_selling_price(data):
    input_df = pd.DataFrame(data, index=[0])
    input_df_scaled = scaler.transform(input_df)
    selling_price_pred = reg_model.predict(input_df_scaled)
    return selling_price_pred[0]

def predict_status(data):
    input_df = pd.DataFrame(data, index=[0])
    status_pred = cls_model.predict(input_df)
    if status_pred[0] == 0:
        return 'LOST'
    else:
        return 'WON'

# Button to make predictions
if st.sidebar.button('Predict'):
    # Predict Selling Price
    selling_price_prediction = predict_selling_price(input_data)
    st.sidebar.write(f'Predicted Selling Price: {selling_price_prediction}')

    # Predict Status
    status_prediction = predict_status(input_data)
    st.sidebar.write(f'Predicted Status: {status_prediction}')
