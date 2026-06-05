import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import joblib

df = pd.read_csv("data/sensor_telemetry_engineered.csv")

feature_cols = [
    "temperature", "vibration", "pressure", "rotational_speed",
    "temperature_rolling_mean", "temperature_rolling_std",
    "vibration_rolling_mean", "vibration_rolling_std",
    "pressure_rolling_mean", "pressure_rolling_std",
    "high_vibration",
]
target_col = "failure"

X = df[feature_cols].copy()
y = df[target_col]

X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, random_state=42)),
    ]),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
}

results = []
best_model = None
best_f1 = -1

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    acc = report["accuracy"]
    prec = report["macro avg"]["precision"]
    rec = report["macro avg"]["recall"]
    f1 = report["macro avg"]["f1-score"]

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc,
    })

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"  ROC-AUC: {auc:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = (name, model)

results_df = pd.DataFrame(results).set_index("Model")
print(f"\n{'='*60}")
print("  Model Comparison")
print(f"{'='*60}")
print(results_df.to_string(float_format="%.4f"))

print(f"\nBest model by F1 score: {best_model[0]} (F1 = {best_f1:.4f})")
joblib.dump(best_model[1], "models/best_model.pkl")
print("Saved best model to models/best_model.pkl")
