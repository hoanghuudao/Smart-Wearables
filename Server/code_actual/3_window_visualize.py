import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configuration Parameters
WINDOW_LENGTH = 50             # number of time steps in each sliding window
WINDOW_STEP_SIZE = 25          # step size between sliding windows
NUM_CHANNELS = 4               # number of sensor channels per sample

# Paths to output directories
TRAIN_DIR = "./3_training_data"
VAL_DIR = "./3_validating_data"
TEST_DIR = "./3_testing_data"
MERGED_DIR = "./2_merged_data"

# Utility Functions
def create_directories():
    for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(directory, exist_ok=True)

def save_windows_as_npy(data, save_path, gesture_name):
    gesture_folder = os.path.join(save_path, gesture_name)
    np.save(os.path.join(save_path, f"{gesture_name}_windows.npy"), data)

# Radar chart replacing scatterplot
def plot_radar_chart(features, labels, label_encoder):
    num_features = features.shape[1]
    num_channels = 4
    stat_names = ['mean', 'std', 'min', 'max']
    feature_labels = [f'{stat}_S{i+1}' for i in range(num_channels) for stat in stat_names]

    label_names = label_encoder.inverse_transform(np.unique(labels))
    avg_per_gesture = {}

    for i, label in enumerate(np.unique(labels)):
        gesture_features = features[labels == label]
        avg_per_gesture[label_names[i]] = np.mean(gesture_features, axis=0)

    angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for gesture_name, values in avg_per_gesture.items():
        values = values.tolist()
        values += values[:1]
        ax.plot(angles, values, label=gesture_name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), feature_labels, fontsize=8)
    plt.title('Radar Chart of Gesture Feature Averages', size=14)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.show()

def plot_pca(features, labels):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Features')
    unique_labels = np.unique(labels)
    legend_labels = [f'gesture{label + 1}' for label in unique_labels]
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Gestures")
    plt.show()

def plot_tsne(features, labels):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate='auto', init='pca', random_state=42)
    tsne_result = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.title('t-SNE of Features')
    unique_labels = np.unique(labels)
    legend_labels = [f'gesture{label + 1}' for label in unique_labels]
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Gestures")
    plt.tight_layout()
    plt.show()

def plot_parallel_coordinates_by_sensors(windows_by_label):
    rows = []
    for gesture_name, windows in windows_by_label.items():
        sensor_means = np.mean(windows, axis=(0, 1))
        row = {f'Sensor{i+1}': sensor_means[i] for i in range(NUM_CHANNELS)}
        row['Gesture'] = gesture_name
        rows.append(row)

    df = pd.DataFrame(rows)
    plt.figure(figsize=(10, 6))
    pd.plotting.parallel_coordinates(df, 'Gesture', colormap='tab10')
    plt.title('Average Sensor Values Per Gesture')
    plt.ylabel('Sensor Reading')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def extract_temp_features(windows):
    features = []
    for window in windows:
        feature_vector = []
        for i in range(window.shape[1]):
            channel_data = window[:, i]
            feature_vector.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data)
            ])
        features.append(feature_vector)
    return np.array(features)

def process_csv_files_by_category(category, save_dir, collect_for_visual=False):
    file_pattern = os.path.join(MERGED_DIR, f"gesture*_*.csv")
    files = glob.glob(file_pattern)

    all_windows = []
    all_labels = []
    windows_by_label = {}

    for file in files:
        basename = os.path.basename(file)
        if f"_{category}.csv" not in basename:
            continue
        gesture_name = basename.split("_")[0].strip()
        try:
            data = np.loadtxt(file, delimiter=",")
        except Exception as e:
            print(f"Failed to load {file}: {e}")
            continue

        num_samples = data.shape[0]
        num_windows = (num_samples - WINDOW_LENGTH) // WINDOW_STEP_SIZE + 1
        if num_windows <= 0:
            continue

        sliding_windows = np.zeros((num_windows, WINDOW_LENGTH, NUM_CHANNELS))
        for i in range(num_windows):
            start = i * WINDOW_STEP_SIZE
            sliding_windows[i, :, :] = data[start:start + WINDOW_LENGTH]

        save_windows_as_npy(sliding_windows, save_dir, gesture_name)

        if collect_for_visual:
            labels = np.full(sliding_windows.shape[0], gesture_name)
            all_windows.append(sliding_windows)
            all_labels.append(labels)
            windows_by_label[gesture_name] = sliding_windows

    if collect_for_visual and all_windows:
        all_windows = np.vstack(all_windows)
        all_labels = np.concatenate(all_labels)
        le = LabelEncoder()
        all_labels_encoded = le.fit_transform(all_labels)

        features = extract_temp_features(all_windows)

        plot_radar_chart(features, all_labels_encoded, le)
        plot_pca(features, all_labels_encoded)
        plot_tsne(features, all_labels_encoded)
        plot_parallel_coordinates_by_sensors(windows_by_label)

def main():
    create_directories()

    print("Processing training data...")
    process_csv_files_by_category("training", TRAIN_DIR, collect_for_visual=True)

    print("Processing validating data...")
    process_csv_files_by_category("validating", VAL_DIR)

    print("Processing testing data...")
    process_csv_files_by_category("testing", TEST_DIR)

if __name__ == "__main__":
    main()
