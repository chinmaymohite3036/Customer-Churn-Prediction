import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, ConfusionMatrixDisplay)
from sklearn.model_selection import cross_val_score

#  Loading Dataset
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

# Pipeline
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

# Compare Training Models
#1) Logistic Regression Classifier

log_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

print("Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)


#2) Decision tree Classifier
tree_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        random_state=42
    ))
])

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)

print(classification_report(
    y_test,
    tree_pred
))

#3) Random Forest Classifier
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42
    ))
])

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print(classification_report(
    y_test,
    rf_pred
))

# 10-fold Cross Validation 
#1) Logistic Regression
log_scores = cross_val_score(
    log_model,
    X_train,
    y_train,
    cv=10,
    scoring="f1_macro"
)

print(log_scores.mean())

#2) Decision Tree
tree_scores = cross_val_score(
    tree_model,
    X_train,
    y_train,
    cv=10,
    scoring="f1_macro"
)

print(tree_scores.mean())

#3) Random Forest   
rf_scores = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv=10,
    scoring="f1_macro"
)

print(rf_scores.mean())

# Create result table 
log_f1 = log_scores.mean()
tree_f1 = tree_scores.mean()
rf_f1 = rf_scores.mean()

results = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "F1 Score": [
        log_f1,
        tree_f1,
        rf_f1
    ]
}
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)
print(results_df)

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

# Comparison Graph
plt.figure(figsize=(8, 6))

plt.bar(
    results_df["Model"],
    results_df["F1 Score"],
    color=["blue", "green", "red"]
)
plt.title("Model Comparison")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.tight_layout()

plt.savefig(
    "screenshots/model_comparison.png"
)

