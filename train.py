import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

 # Loading Dataset
df = pd.read_csv("data/churn.csv")

# Data Cleaning
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
df.dropna(inplace=True)
df = df.drop("customerID", axis=1)

# Seperate Features and Labels
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-Test Split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# List numerical and categorical data
num_cols = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

cat_cols = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

#Pipeline
# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Full pipepline
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Selected Logistic Regression model 
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",
     LogisticRegression(
         max_iter=1000
     ))
])

model_path = "models/churn_model.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    # Train model
    model.fit(
        X_train,
        y_train
    )
    print("Model trained successfully!")

# Evaluate
y_pred = model.predict(X_test)

print("Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

if not os.path.exists(model_path):
    # Create model folder
    os.makedirs(
        "models",
        exist_ok=True
    )

    # Save Model
    joblib.dump(
        model,
        model_path
    )

    print("Model saved successfully!")

# Load saved model

loaded_model = joblib.load(model_path)

print("Model loaded successfully!")

# Infernce

sample_customer = X_test.iloc[[0]]

prediction = loaded_model.predict(
    sample_customer
)

actual = y_test.iloc[0]

print("Actual:", actual)
print("Predicted:", prediction[0])

# Custom customer
custom_customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.50,
    "TotalCharges": 477.50
}])

prediction = loaded_model.predict(custom_customer)

print("Prediction:", prediction[0])

if prediction[0] == "Yes":
    print("Customer is likely to churn.")
else:
    print("Customer is likely to stay.")


# Second Customer

custom_customer_2 = pd.DataFrame([{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 65.00,
    "TotalCharges": 3900.00
}])

prediction = loaded_model.predict(custom_customer_2)

print("Prediction:", prediction[0])

if prediction[0] == "Yes":
    print("Customer is likely to churn.")
else:
    print("Customer is likely to stay.")
