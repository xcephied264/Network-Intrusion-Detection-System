import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import GridSearchCV

# 1. Load dataset
train = pd.read_csv("kdd_train.csv")
test = pd.read_csv("kdd_test.csv")

# 2. Convert labels
# normal = 0, attack = 1
def label_map(label):
    return 0 if str(label).strip() == "normal" else 1

train['labels'] = train['labels'].apply(label_map)
test['labels'] = test['labels'].apply(label_map)

# 3. Split features / target
X_train = train.drop('labels', axis=1)
y_train = train['labels']

X_test = test.drop('labels', axis=1)
y_test = test['labels']

# 4. One-hot encoding
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Align test columns to train columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 5. Scale data for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Helper function
def print_metrics(model_name, y_true, y_pred):
    print(f"\n{'='*50}")
    print(f"{model_name} Results")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"]))
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-score :", f1_score(y_true, y_pred))

# 7. Fine-tuning Linear SVM
print("\nTuning Linear SVM...")

svm_param_grid = {
    'C': [0.1, 1, 10],
    'class_weight': [None, 'balanced']
}

svm_grid = GridSearchCV(
    estimator=LinearSVC(max_iter=10000, random_state=42),
    param_grid=svm_param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=1
)

svm_grid.fit(X_train_scaled, y_train)

best_svm = svm_grid.best_estimator_
y_pred_svm = best_svm.predict(X_test_scaled)

print("\nBest SVM Parameters:", svm_grid.best_params_)
print("Best SVM CV F1-score:", svm_grid.best_score_)

# 8. Fine-tuning Random Forest
print("\nTuning Random Forest...")

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': [None, 'balanced']
}

rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print("\nBest Random Forest Parameters:", rf_grid.best_params_)
print("Best Random Forest CV F1-score:", rf_grid.best_score_)

# 9. Print final metrics
print_metrics("Fine-tuned Linear SVM", y_test, y_pred_svm)
print_metrics("Fine-tuned Random Forest", y_test, y_pred_rf)

# 10. Confusion Matrix - SVM
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_svm, display_labels=["Normal", "Attack"]
)
plt.title("Fine-tuned Linear SVM Confusion Matrix")
plt.show()

# 11. Confusion Matrix - RF
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_rf, display_labels=["Normal", "Attack"]
)
plt.title("Fine-tuned Random Forest Confusion Matrix")
plt.show()