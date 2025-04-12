import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import glob
import re
import joblib
from scipy.stats import skew, kurtosis, iqr
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Settings
GESTURES = ["gesture1", "gesture6"]
SENSOR_2 = 1
SENSOR_3 = 2

def plot_confusion(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
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

    train_mask = np.isin(y_train, GESTURES)
    val_mask = np.isin(y_val, GESTURES)
    test_mask = np.isin(y_test, GESTURES)

    x_train = x_train[train_mask]
    y_train = y_train[train_mask]
    x_val = x_val[val_mask]
    y_val = y_val[val_mask]
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    x_train = extract_emphasized_sensor_features(x_train, SENSOR_2, SENSOR_3)
    x_val = extract_emphasized_sensor_features(x_val, SENSOR_2, SENSOR_3)
    x_test = extract_emphasized_sensor_features(x_test, SENSOR_2, SENSOR_3)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    model = SVC(C=0.1, gamma=0.1, kernel='rbf', probability=True)
    model_name = "SVM"
    print(f"\nTraining {model_name} on gesture1 vs gesture6...\n")
    model.fit(x_train, y_train_enc)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    val_acc = accuracy_score(y_val_enc, val_pred)
    test_acc = accuracy_score(y_test_enc, test_pred)
    balance_score = (val_acc + test_acc) / 2

    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Balanced Score: {balance_score:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test_enc, test_pred, target_names=label_encoder.classes_))
    plot_confusion(y_test_enc, test_pred, label_encoder.classes_, model_name)

    out_dir = "./4.3_model_gesture_1_6"
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "model.pkl"))
    joblib.dump(label_encoder, os.path.join(out_dir, "encoder.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    print(f"\nSaved {model_name} model to {out_dir}")

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
    label = match.group(0)
    for window in windows:
        feature_vector = []
        for sensor in range(window.shape[1]):
            signal = window[:, sensor]
            feature_vector.extend([
                np.min(signal), np.max(signal), np.mean(signal), np.median(signal),
                np.std(signal), np.ptp(signal), np.sqrt(np.mean(signal ** 2)),
                skew(signal), kurtosis(signal), iqr(signal)
            ])
        features.append(feature_vector)
        labels.append(label)
    return np.array(features), np.array(labels)

def extract_emphasized_sensor_features(X, sensor2_index, sensor3_index):
    num_features_per_sensor = 10
    f2 = X[:, sensor2_index * num_features_per_sensor: (sensor2_index + 1) * num_features_per_sensor]
    f3 = X[:, sensor3_index * num_features_per_sensor: (sensor3_index + 1) * num_features_per_sensor]
    return np.hstack([f2, f3, f3])

if __name__ == "__main__":
    main()
