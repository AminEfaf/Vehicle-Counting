# Vehicle Counting

This project was developed as an **Image Processing course project**. It demonstrates how image preprocessing techniques can improve the quality of traffic video frames before vehicle detection and counting.

The project uses **YOLOv11** for vehicle detection and tracking, while the main focus is on the preprocessing pipeline applied to each frame.

## Project structure

```text
vehicle-counting/
├── VehicleCounting.py
├── requirements.txt
├── README.md
└── videos/
    ├── night.mp4
    ├── onewayup.mp4
    ├── TestVideo2.mp4
    └── TestVideo3.mp4
```

## Image processing pipeline

The following preprocessing techniques are applied sequentially to each video frame before detection:

* Normalization
* CLAHE (Contrast Limited Adaptive Histogram Equalization)
* Bilateral filtering for noise reduction
* Edge enhancement
* Shadow removal
* Logarithmic transformation
* Unsharp masking
* Edge blending

These operations help improve contrast, reduce noise, enhance edges, and make vehicle features more visible under different lighting conditions.

## How to Run

1. Clone the repository:

```bash
git clone <repository-url>
cd vehicle-counting
```

2. Install the required Python libraries:

```bash
pip install -r requirements.txt
```

3. Make sure the video files are inside the `videos/` folder:

```text
videos/
├── night.mp4
├── onewayup.mp4
├── TestVideo2.mp4
└── TestVideo3.mp4
```

4. Run the script:

```bash
python VehicleCounting.py
```

5. A GUI window will appear. Select one of the available videos to start processing.

6. The program will display the processed video with vehicle detection, tracking, directional counting, and accuracy results.

## Output

The script provides:

* Real-time vehicle detection and tracking.
* Upward and downward vehicle counts.
* Total detected vehicles compared with actual counts.
* Accuracy charts for overall detection and each vehicle type.

## Collaborators

- [Mohammad Amin Efaf](https://github.com/aminefaf)  
- [Arman Ghorbanpour](https://github.com/armanghorbanpour)
