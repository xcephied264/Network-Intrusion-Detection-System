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

# 1. Load dataset
train = pd.read_csv("kdd_train.csv")
test = pd.read_csv("kdd_test.csv")

# 2. Convert labels: normal = 0, attack = 1
def label_map(label):
    return 0 if str(label).strip() == "normal" else 1

train['labels'] = train['labels'].apply(label_map)
test['labels'] = test['labels'].apply(label_map)

# 3. Split features and target
X_train = train.drop('labels', axis=1)
y_train = train['labels']

X_test = test.drop('labels', axis=1)
y_test = test['labels']

# 4. One-hot encoding
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# 5. Align columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 6. Scale for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Linear SVM
svm = LinearSVC(max_iter=5000, random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

# 8. Random Forest
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 9. Results function
def print_metrics(model_name, y_true, y_pred):
    print(f"\n{model_name} Results")
    print(classification_report(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1-score:", f1_score(y_true, y_pred))

print_metrics("Linear SVM", y_test, y_pred_svm)
print_metrics("Random Forest", y_test, y_pred_rf)

# 10. Confusion Matrix - SVM
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_svm, display_labels=["Normal", "Attack"]
)
plt.title("Linear SVM Confusion Matrix")
plt.show()

# 11. Confusion Matrix - Random Forest
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_rf, display_labels=["Normal", "Attack"]
)
plt.title("Random Forest Confusion Matrix")
plt.show()