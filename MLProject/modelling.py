import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load data
X_train = pd.read_csv(
    "heart_preprocessing/X_train.csv"
)

X_test = pd.read_csv(
    "heart_preprocessing/X_test.csv"
)

y_train = pd.read_csv(
    "heart_preprocessing/y_train.csv"
)

y_test = pd.read_csv(
    "heart_preprocessing/y_test.csv"
)

# MLflow
mlflow.set_experiment("Heart_Disease_Experiment")
mlflow.autolog()

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Training
model.fit(X_train, y_train.values.ravel())

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Logging parameter
mlflow.log_param("n_estimators", 100)

# Logging metric
mlflow.log_metric("accuracy", accuracy)

# Folder artifacts
os.makedirs("artifacts", exist_ok=True)

# Save model
joblib.dump(
    model,
    "artifacts/random_forest_model.pkl"
)

# Log artifact model
mlflow.log_artifact(
    "artifacts/random_forest_model.pkl"
)

# Classification report
report = classification_report(
    y_test,
    y_pred
)

with open(
    "artifacts/classification_report.txt",
    "w"
) as f:
    f.write(report)

mlflow.log_artifact(
    "artifacts/classification_report.txt"
)

# Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred
)

with open(
    "artifacts/confusion_matrix.txt",
    "w"
) as f:
    f.write(str(cm))

mlflow.log_artifact(
    "artifacts/confusion_matrix.txt"
)

# Log model ke MLflow
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="random_forest_model"
)

print("Training selesai")
print("Accuracy:", accuracy)