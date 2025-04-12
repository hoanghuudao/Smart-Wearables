import socket
import numpy as np
import joblib
from collections import deque
from scipy.stats import skew, kurtosis, iqr

# Load model components
unknown_known_model = joblib.load("./4.1_model_known_unknown/model.pkl")
unknown_known_scaler = joblib.load("./4.1_model_known_unknown/scaler.pkl")

group_model = joblib.load("./4.2_model_gesture_classifier/model.pkl")
group_scaler = joblib.load("./4.2_model_gesture_classifier/scaler.pkl")
group_encoder = joblib.load("./4.2_model_gesture_classifier/encoder.pkl")

binary_models = {
    "gesture1or6": {
        "model": joblib.load("./4.3_model_gesture_1_6/model.pkl"),
        "encoder": joblib.load("./4.3_model_gesture_1_6/encoder.pkl"),
        "scaler": joblib.load("./4.3_model_gesture_1_6/scaler.pkl")
    },
    "gesture2or3": {
        "model": joblib.load("./4.3_model_gesture_2_3/model.pkl"),
        "encoder": joblib.load("./4.3_model_gesture_2_3/encoder.pkl"),
        "scaler": joblib.load("./4.3_model_gesture_2_3/scaler.pkl")
    },
    "gesture4or5": {
        "model": joblib.load("./4.3_model_gesture_4_5/model.pkl"),
        "encoder": joblib.load("./4.3_model_gesture_4_5/encoder.pkl"),
        "scaler": joblib.load("./4.3_model_gesture_4_5/scaler.pkl")
    }
}

gesture_actions = {
    "gesture1": "Change Audio Mode",
    "gesture2": "Activate Voice Assistant",
    "gesture3": "React Heart",
    "gesture4": "Make a Call",
    "gesture5": "Confirm / Yes",
    "gesture6": "View Notifications",
    "gesture7": "Play/Pause Music",
    "gesture8": "Reject / No",
    "unknown": "Unknown Action"
}

# Sensor window buffer
window_size = 50
sensor_window = deque(maxlen=window_size)


def extract_features(window):
    features = []
    window = np.array(window)  # shape: (50, 4)
    for i in range(window.shape[1]):
        signal = window[:, i]
        features.extend([
            np.min(signal), np.max(signal), np.mean(signal), np.median(signal),
            np.std(signal), np.ptp(signal), np.sqrt(np.mean(signal ** 2)),
            skew(signal), kurtosis(signal), iqr(signal),
        ])
    return np.array(features)


def predict_gesture(window):
    features = extract_features(window)

    # Step 1: Known vs Unknown
    is_known = unknown_known_model.predict(unknown_known_scaler.transform([features]))[0] == 1
    if not is_known:
        return "unknown", gesture_actions["unknown"]

    # Step 2: Gesture group
    group_label = group_model.predict(group_scaler.transform([features]))[0]
    group = group_encoder.inverse_transform([group_label])[0]

    if group in ["gesture7", "gesture8"]:
        return group, gesture_actions[group]

    # Step 3: Binary classification with synchronized slicing
    binary = binary_models.get(group)
    if binary is None:
        return "unknown", gesture_actions["unknown"]

    if group in ["gesture1or6", "gesture2or3"]:
        s2 = features[10:20]
        s3 = features[20:30]
        sliced = np.hstack([s2, s3, s3])  # 30 features
    elif group == "gesture4or5":
        sliced = features[30:40]  # 10 features
    else:
        sliced = features  # fallback

    final_label = binary["model"].predict(binary["scaler"].transform([sliced]))[0]
    gesture = binary["encoder"].inverse_transform([final_label])[0]

    return gesture, gesture_actions.get(gesture, "Unknown Action")


# Server settings
HOST = "0.0.0.0"
PORT = 8080


def main():
    print(f"Server listening on {HOST}:{PORT}...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)

    conn, addr = server.accept()
    print(f"Connected to Arduino at {addr}")

    buffer = ""

    while True:
        try:
            # Use ASCII encoding and ignore errors
            data = conn.recv(1024).decode('ascii', errors="ignore")
            if not data:
                continue

            buffer += data
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                # More thorough cleaning
                line = ''.join(c for c in line if c.isdigit() or c == ',')

                parts = line.split(",")
                if len(parts) != 4:
                    print(f"Invalid line (wrong number of values): {line}")
                    continue

                try:
                    values = []
                    for p in parts:
                        cleaned = ''.join(c for c in p if c.isdigit())
                        if cleaned:
                            values.append(float(cleaned))
                        else:
                            raise ValueError(f"Empty value after cleaning: '{p}'")

                    if len(values) == 4:
                        sensor_window.append(values)

                        if len(sensor_window) == window_size:
                            gesture, action = predict_gesture(sensor_window)
                            print(f"Gesture predicted: {gesture} → {action} | Sensor Input: {values}")
                    else:
                        print(f"Incomplete values: {values}")

                except ValueError as e:
                    print(f"Invalid line (conversion error): {repr(line)} - {e}")
                    continue

        except ConnectionResetError:
            print("Arduino disconnected. Waiting for reconnect...")
            conn, addr = server.accept()
            print(f"Reconnected: {addr}")


if __name__ == "__main__":
    main()
