# Vehicle Counting System

This project focuses on detecting and counting vehicles in various conditions, such as day and night, using the YOLOv11 model. The system integrates real-time object detection, tracking, and pre-processing techniques to improve accuracy.

---

## Project Overview

The Vehicle Counting System provides:
- **Real-time vehicle detection** using YOLOv11.
- **Pre-processing techniques** to enhance image quality.
- **Tracking system** for counting vehicle movement in different directions.
- **Accuracy evaluation** by comparing detected counts with actual values.

---

## Features

1. **YOLOv11 Model**
   - Detects cars, motorcycles, buses, and trucks.
2. **Pre-Processing Pipeline**
   - Normalization, contrast enhancement, noise reduction, and shadow removal.
3. **Real-Time Tracking**
   - Identifies vehicle movement and prevents duplicate counts.
4. **Accuracy Measurement**
   - Compares detected counts with actual vehicle numbers.
5. **Data Visualization**
   - Generates bar charts showing detection accuracy.

---

## Installation & Execution

### Prerequisites
- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Ultralytics YOLO library

### Steps
1. Clone the repository and extract the files.
2. Open the project in **Visual Studio Code**.
3. Run `Complete_Code.py`.
4. Select a video from the GUI.
5. View real-time vehicle tracking and final accuracy results.

---

## Results & Visualization

### Accuracy Calculation
The system computes accuracy by:
- **Comparing actual vs. detected vehicle counts.**
- **Measuring detection accuracy for each vehicle type.**

### Graphical Representation
Two bar charts illustrate:
- **Total detected vs. actual vehicles.**
- **Accuracy percentage per vehicle type.**

---

## Collaborators
- **Mohammad Amin Efaf**
- **Arman GhorbanPour**

## Feedback
We’d love to hear your thoughts and suggestions! Feel free to reach out or open an issue in this repository.

---
