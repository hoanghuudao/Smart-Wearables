import socket
import csv
import datetime
import os
import re

# Server settings
HOST = "0.0.0.0"  # Listens on all available interfaces
PORT = 8080  # Must match Arduino's serverPort

# Create a directory to store training data if it doesn't exist
data_dir = "../code_actual/1_saving_data"
os.makedirs(data_dir, exist_ok=True)

# Generate a new CSV file per session
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = os.path.join(data_dir, f"training_sensor_data_{timestamp_str}.csv")


def main():
    # Create and open CSV file for writing data
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Gesture", "Sensor1", "Sensor2", "Sensor3", "Sensor4"])

        # Set up the server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)

        print(f"Listening for connections on port {PORT}...")

        # Accept connection
        client_socket, client_address = server_socket.accept()
        print(f"Connected to {client_address}")

        buffer = ""
        line_count = 0

        try:
            while True:
                # Receive data
                data = client_socket.recv(4096).decode("utf-8")
                if not data:
                    print("Client disconnected")
                    break

                # Add to buffer
                buffer += data

                # Special handling for gesture pattern
                # This regex finds "gesture\d," pattern which starts a new line
                split_data = re.split(r'(gesture\d,)', buffer)

                if len(split_data) > 1:
                    # Reconstructed lines list
                    processed_lines = []

                    # First element might be an incomplete line or empty
                    current = split_data[0]

                    # Process all gesture markers and following data
                    for i in range(1, len(split_data), 2):
                        if i + 1 < len(split_data):
                            # Complete line: gesture marker + data
                            processed_lines.append(split_data[i] + split_data[i + 1])
                        else:
                            # Last gesture marker might not have complete data
                            current = split_data[i]

                    # Process complete lines
                    for line in processed_lines:
                        line = line.strip()
                        if not line:
                            continue

                        # Fix the issue with repeated gesture name at the end
                        if "gesture" in line[line.find(',') + 1:]:
                            line = line[:line.rfind("gesture")]

                        # Skip if it still doesn't look valid
                        parts = line.split(',')
                        if len(parts) != 5:
                            print(f"Invalid line (parts={len(parts)}): {line}")
                            continue

                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        gesture = parts[0]
                        sensor_values = parts[1:]

                        # Write to file
                        writer.writerow([timestamp, gesture] + sensor_values)
                        file.flush()
                        line_count += 1

                        if line_count % 10 == 0:
                            print(f"Processed {line_count} lines")

                    # Keep only the unprocessed portion
                    buffer = current

        except KeyboardInterrupt:
            print("Server stopped.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            print(f"Total lines processed: {line_count}")
            client_socket.close()
            server_socket.close()


if __name__ == "__main__":
    main()
