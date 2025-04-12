import os
import pandas as pd

def load_and_merge_csv(folder_path):
    # Create output directory
    output_dir = "./2_merged_data"
    os.makedirs(output_dir, exist_ok=True)

    # Define person groups
    training_people = {"dao", "vaino", "kiet"}
    validating_people = {"daniyar"}
    testing_people = {"linh", "nimrod"}

    # Gather all CSV files
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Prepare grouped data
    grouped_data = {
        "training": {},
        "validating": {},
        "testing": {}
    }

    for file in all_files:
        name = os.path.splitext(file)[0].lower()  # filename without extension
        filepath = os.path.join(folder_path, file)

        try:
            df = pd.read_csv(filepath)
            required_columns = {'Gesture', 'Sensor1', 'Sensor2', 'Sensor3', 'Sensor4'}
            if not required_columns.issubset(df.columns):
                print(f"Skipping {file}: Missing required columns")
                continue

            # Determine dataset category
            if name in training_people:
                category = "training"
            elif name in validating_people:
                category = "validating"
            elif name in testing_people:
                category = "testing"
            else:
                print(f"Skipping {file}: Unrecognized person name '{name}'")
                continue

            # Append gesture-wise data
            for gesture in df['Gesture'].unique():
                gesture_df = df[df['Gesture'] == gesture][['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4']]
                if gesture not in grouped_data[category]:
                    grouped_data[category][gesture] = []
                grouped_data[category][gesture].append(gesture_df)

        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

    # Save merged datasets
    for category, gesture_dict in grouped_data.items():
        for gesture, dfs in gesture_dict.items():
            merged_df = pd.concat(dfs, ignore_index=True).drop_duplicates().ffill()
            output_path = os.path.join(output_dir, f"{gesture}_{category}.csv")
            merged_df.to_csv(output_path, index=False, header=False)
            print(f"Saved: {output_path}")

def main():
    folder_path = input("Enter the path to the folder containing gesture CSV files: ")
    load_and_merge_csv(folder_path)

if __name__ == "__main__":
    main()
