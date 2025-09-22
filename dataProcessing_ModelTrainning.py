import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv(r"C:\Users\rutur\OneDrive\Documents\5TH_ML\cardio_train.csv")

# Data Preprocessing
# Handle outliers using IQR
Q1 = data[['ap_hi', 'ap_lo', 'height', 'weight']].quantile(0.25)
Q3 = data[['ap_hi', 'ap_lo', 'height', 'weight']].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data[['ap_hi', 'ap_lo', 'height', 'weight']] < (Q1 - 1.5 * IQR)) |
              (data[['ap_hi', 'ap_lo', 'height', 'weight']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Drop missing values
data.dropna(inplace=True)

# Feature engineering
data['age'] = data['age'] / 365.25  # Convert age to years
data['bmi'] = data['weight'] / (data['height'] / 100) ** 2  # Calculate BMI
data['gender'] = data['gender'].map({1: 0, 2: 1})  # 0=male, 1=female
data.drop(['weight', 'id'], axis=1, inplace=True)

# Feature scaling
scaler = StandardScaler()
numerical_cols = ['age', 'height', 'ap_hi', 'ap_lo', 'bmi']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Split data
X = data.drop('cardio', axis=1)
y = data['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Model Training and Evaluation
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
}

# Train and evaluate models
best_model = None
best_accuracy = 0
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"\nBest Random Forest Parameters: {grid_search.best_params_}")
print(f"Best Random Forest Accuracy: {grid_search.best_score_:.2f}")

# Select best model
if grid_search.best_score_ > best_accuracy:
    best_model = grid_search.best_estimator_

# Evaluate best model
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nBest Model ROC-AUC: {auc:.2f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Save model and scaler
joblib.dump(best_model, 'cardio_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Prediction function
def predict_disease(user_input):
    user_df = pd.DataFrame([user_input])
    user_df['age'] = user_df['age'] * 365.25  # Convert to days
    user_df['bmi'] = user_df['weight'] / (user_df['height'] / 100) ** 2
    user_df['gender'] = user_df['gender'].map({'Male': 0, 'Female': 1})
    user_df[numerical_cols] = scaler.transform(user_df[numerical_cols])
    user_df.drop(['weight'], axis=1, inplace=True)
    user_df = user_df.reindex(columns=X_train.columns, fill_value=0)
    prediction = best_model.predict(user_df)
    probability = best_model.predict_proba(user_df)[:, 1]
    return 'Cardiovascular Disease' if prediction[0] == 1 else 'No Cardiovascular Disease', probability[0]