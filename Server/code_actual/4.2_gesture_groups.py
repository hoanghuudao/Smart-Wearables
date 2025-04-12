import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import glob
import re
import joblib
from scipy.stats import skew, kurtosis, iqr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Mapping individual gestures to grouped categories
GROUP_MAPPING = {
    "gesture1": "gesture1or6",
    "gesture6": "gesture1or6",
    "gesture2": "gesture2or3",
    "gesture3": "gesture2or3",
    "gesture4": "gesture4or5",
    "gesture5": "gesture4or5",
    "gesture7": "gesture7",
    "gesture8": "gesture8"
}

def plot_confusion(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {title}')
    plt.tight_layout()
    plt.show()

def main():
    x_train, y_train = load_and_extract(glob.glob("./3_training_data/*.npy"))
    x_val, y_val = load_and_extract(glob.glob("./3_validating_data/*.npy"))
    x_test, y_test = load_and_extract(glob.glob("./3_testing_data/*.npy"))

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    model_name = "Random Forest"
    model = RandomForestClassifier(n_estimators=50, max_depth=None)

    print(f"\nTraining {model_name} for gesture group classification...\n")
    model.fit(x_train, y_train_enc)

    val_pred = model.predict(x_val)
    val_acc = accuracy_score(y_val_enc, val_pred)
    print(f"Validation Accuracy: {val_acc:.4f}")

    test_pred = model.predict(x_test)
    test_acc = accuracy_score(y_test_enc, test_pred)
    print(f"Test Accuracy: {test_acc:.4f}")

    balanced_score = (val_acc + test_acc) / 2
    print(f"Balanced Score: {balanced_score:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test_enc, test_pred, target_names=label_encoder.classes_, zero_division=0))
    plot_confusion(y_test_enc, test_pred, label_encoder.classes_, model_name)

    os.makedirs("./4.2_model_gesture_classifier", exist_ok=True)
    joblib.dump(model, "4.2_model_gesture_classifier/model.pkl")
    joblib.dump(scaler, "./4.2_model_gesture_classifier/scaler.pkl")
    joblib.dump(label_encoder, "4.2_model_gesture_classifier/encoder.pkl")
    print(f"\nSaved {model_name} and preprocessing tools to ./4.2_model_gesture_classifier")

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
        raise ValueError(f"Invalid filename: {file}")
    original_label = match.group(0)
    grouped_label = GROUP_MAPPING.get(original_label)
    if not grouped_label:
        raise ValueError(f"Unknown gesture label: {original_label}")

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
        labels.append(grouped_label)

    return np.array(features), np.array(labels)

if __name__ == "__main__":
    main()
