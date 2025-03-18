import cv2
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt
from tkinter import Tk, Button
import numpy as np


# Load the YOLO model
model = YOLO('yolo11l.pt')
class_list = model.names

# Initialize variables for video path and parameters
video_path = ''
line_start = (0, 0)
line_end = (0, 0)
actual_counts = {}

# Define preprocessing functions
def normalize_image(image):
    image = image.astype('float32') / 255.0
    return (image * 255).astype('uint8')

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def reduce_noise(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def enhance_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    return cv2.addWeighted(gray, 2.0, gaussian, -1.0, 0)

def remove_shadows(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    median = cv2.medianBlur(l, 5)
    lab = cv2.merge([median, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def log_transform(image):
    # Convert image to float32 to prevent overflow during log transformation
    c = 255 / (np.log(1 + np.max(image)))
    log_image = c * (np.log(1 + image.astype('float32')))
    return np.clip(log_image, 0, 255).astype('uint8')

def unsharp_mask(image, strength=1.5, kernel_size=(5, 5)):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    # Create the sharpened image
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return sharpened

def preprocess_frame(frame):
    # Sequential preprocessing
    image = normalize_image(frame)
    image = apply_clahe(image)
    image = reduce_noise(image)
    edge_map = enhance_edges(image)
    image = remove_shadows(image)
    image = log_transform(image)  # Logarithmic transformation
    image = unsharp_mask(image)   # Unsharp masking
    # Blend edge map with processed image
    edge_map_bgr = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
    final_image = cv2.addWeighted(image, 0.7, edge_map_bgr, 0.3, 0)
    return final_image


# Define a function to set parameters based on video selection
def set_parameters(video):
    global video_path, line_start, line_end, actual_counts

    if video == "night":
        video_path = 'videos/night.mp4'
        line_start = (200, 425)
        line_end = (1300, 425)
        actual_counts = {
            "car": 522,
            "truck": 6,
            "bus": 2,
            "motorcycle": 1
        }
    elif video == "guilan":
        video_path = 'videos/onewayup.mp4'
        line_start = (0, 300)
        line_end = (1220, 300)
        actual_counts = {
            "car": 105,
            "truck": 7,
            "bus": 4,
            "motorcycle": 2
        }
    elif video == "TestVideo_2":
        video_path = 'videos/TestVideo2.mp4'
        line_start = (0, 430)
        line_end = (1400, 430)
        actual_counts = {
            "car": 289,
            "truck": 14,
            "bus": 6,
            "motorcycle": 1
        }
    elif video == "TestVideo_3":
        video_path = 'videos/TestVideo3.mp4'
        line_start = (350, 500)
        line_end = (1500, 500)
        actual_counts = {
            "car": 29,
            "motorcycle": 1
        }

    # Close the GUI window and start processing
    root.destroy()

# Create the GUI window
root = Tk()
root.title("Select Video")

Button(root, text="Night", command=lambda: set_parameters("night"), width=20, height=2).pack(pady=10)
Button(root, text="Guilan", command=lambda: set_parameters("guilan"), width=20, height=2).pack(pady=10)
Button(root, text="TestVideo_2", command=lambda: set_parameters("TestVideo_2"), width=20, height=2).pack(pady=10)
Button(root, text="TestVideo_3", command=lambda: set_parameters("TestVideo_3"), width=20, height=2).pack(pady=10)

root.mainloop()

# Open the video file
cap = cv2.VideoCapture(video_path)

# Calculate slope and intercept of the line
slope = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])
intercept = line_start[1] - slope * line_start[0]

# Dictionaries for tracking objects
class_counts_up = defaultdict(int)  # Vehicles moving upward
class_counts_down = defaultdict(int)  # Vehicles moving downward
crossed_ids_up = set()
crossed_ids_down = set()
object_positions = {}  # To store previous positions of objects

# Calculate total actual vehicles automatically
total_actual_vehicles = sum(actual_counts.values())

while cap.isOpened():
    ret, frame = cap.read()
    frame=preprocess_frame(frame)
    if not ret:
        break

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7])
    # 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
    if results[0].boxes.data is not None:
        boxes = results[0].boxes.xyxy
        track_ids = results[0].boxes.id.int().tolist()
        class_indices = results[0].boxes.cls.int().tolist()

        # Draw the diagonal line
        cv2.line(frame, line_start, line_end, (0, 0, 255), 3)

        for box, track_id, class_idx in zip(boxes, track_ids, class_indices):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())  # Convert to CPU only for drawing
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            class_name = class_list[class_idx]

            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate the y-position of the line at cx
            line_y_at_x = slope * cx + intercept

            # Check if object crosses the line downward
            if track_id in object_positions:
                prev_cx, prev_cy = object_positions[track_id]

                # From top to bottom
                if prev_cy < line_y_at_x <= cy and track_id not in crossed_ids_down:
                    crossed_ids_down.add(track_id)
                    class_counts_down[class_name] += 1

                # From bottom to top
                if prev_cy > line_y_at_x >= cy and track_id not in crossed_ids_up:
                    crossed_ids_up.add(track_id)
                    class_counts_up[class_name] += 1

            # Update the object's position
            object_positions[track_id] = (cx, cy)

        # Display the counts on the frame
        y_offset = 30
        cv2.putText(frame, "Downward Count:", (50, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 30
        for class_name, count in class_counts_down.items():
            cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 30

        y_offset += 20  # Add some spacing
        cv2.putText(frame, "Upward Count:", (50, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_offset += 30
        for class_name, count in class_counts_up.items():
            cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 30

    # Show the frame
    cv2.imshow("YOLO Object Tracking & Counting", frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Calculate total detected vehicles and accuracy
detected_counts = {
    "car": class_counts_up["car"] + class_counts_down["car"],
    "truck": class_counts_up["truck"] + class_counts_down["truck"],
    "bus": class_counts_up["bus"] + class_counts_down["bus"],
    "motorcycle": class_counts_up["motorcycle"] + class_counts_down["motorcycle"]
}

# Calculate accuracy for each type
accuracies = {vehicle: (detected_counts[vehicle] / actual_counts[vehicle]) * 100 
              for vehicle in actual_counts}

# Display overall accuracy
total_detected_vehicles = sum(detected_counts.values())
overall_accuracy = (total_detected_vehicles / total_actual_vehicles) * 100

# Plotting results
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Overall accuracy
axes[0].bar(['Actual Vehicles', 'Detected Vehicles'], [total_actual_vehicles, total_detected_vehicles], color=['blue', 'green'])
axes[0].set_title(f'YOLO Overall Detection Accuracy: {overall_accuracy:.2f}%')
axes[0].set_ylabel('Count')

# Add annotations for overall accuracy
for i, value in enumerate([total_actual_vehicles, total_detected_vehicles]):
    axes[0].text(i, value + 2, str(value), ha='center', fontsize=10)

# Accuracy for each vehicle type
bars = axes[1].bar(accuracies.keys(), accuracies.values(), color='orange')
axes[1].set_title('Accuracy by Vehicle Type')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_ylim(0, 120)

# Add annotations for each vehicle type
for bar, vehicle in zip(bars, accuracies.keys()):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width() / 2, height + 2, f'{height:.2f}%', ha='center', fontsize=10)
    axes[1].text(bar.get_x() + bar.get_width() / 2, -10, f'{detected_counts[vehicle]}', ha='center', fontsize=10, color='black')

plt.tight_layout()
plt.show()

