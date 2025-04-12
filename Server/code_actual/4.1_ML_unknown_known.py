import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import glob
import re
import joblib
from scipy.stats import skew, kurtosis, iqr
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {title}')
    plt.tight_layout()
    plt.show()

def main():
    x_train, _ = load_and_extract(glob.glob("./3_training_data/*.npy"))
    x_val, y_val = load_and_extract(glob.glob("./3_validating_data/*.npy"))
    x_test, y_test = load_and_extract(glob.glob("./3_testing_data/*.npy"))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    model = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
    model_name = "Isolation Forest"

    print(f"\nTraining {model_name} on known vs unknown gestures...\n")
    model.fit(x_train)

    val_pred = model.predict(x_val)
    val_pred = np.where(val_pred == 1, "known", "unknown")
    val_acc = accuracy_score(y_val, val_pred)
    print(f"Validation Accuracy: {val_acc:.4f}")

    test_pred = model.predict(x_test)
    test_pred = np.where(test_pred == 1, "known", "unknown")
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Test Accuracy: {test_acc:.4f}")

    balanced_score = (val_acc + test_acc) / 2
    print(f"Balanced Score: {balanced_score:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test, test_pred, target_names=["known", "unknown"], zero_division=0))
    plot_confusion(y_test, test_pred, ["known", "unknown"], model_name)

    os.makedirs("./4.1_model_known_unknown", exist_ok=True)
    joblib.dump(model, "4.1_model_known_unknown/model.pkl")
    joblib.dump(scaler, "./4.1_model_known_unknown/scaler.pkl")
    print(f"\nSaved {model_name} and scaler to ./4.1_model_known_unknown")

def load_and_extract(file_list):
    x_total, y_total = [], []
    for file in file_list:
        x, y = extract_features(file)
        x_total.append(x)
        y_total.append(y)
    return np.vstack(x_total), np.concatenate(y_total)

def extract_features(file):
    windows = np.load(file)
    features, labels = [], []

    match = re.search(r"gesture\d+", file)
    if not match:
        label = "unknown"
    else:
        gesture = match.group(0)
        label = "known" if gesture in [f"gesture{i}" for i in range(1, 9)] else "unknown"

    for window in windows:
        feature_vector = []
        for sensor in range(window.shape[1]):
            signal = window[:, sensor]
            feature_vector.extend([
                np.min(signal),
                np.max(signal),
                np.mean(signal),
                np.median(signal),
                np.std(signal),
                np.ptp(signal),
                np.sqrt(np.mean(signal**2)),
                skew(signal),
                kurtosis(signal),
                iqr(signal)
            ])
        features.append(feature_vector)
        labels.append(label)

    return np.array(features), np.array(labels)

if __name__ == "__main__":
    main()
