# Multisensor Fusion & Real-Time 3D Object Tracking Perception Pipeline Using KITTI Dataset

## Project Overview

This project implements a multisensor fusion and real-time 3D object tracking perception pipeline using the KITTI dataset. The pipeline fuses LIDAR and camera data to detect and track multiple objects in 3D space. It visualizes projected LIDAR points on camera images, draws 3D bounding boxes, and performs persistent multi-object tracking with a Kalman filter-based tracker.

The system also supports animation and trajectory visualization, providing a comprehensive perception tool ideal for autonomous driving research, robotics, and computer vision applications.

---

## Features

- **Sensor Fusion**: Projects 3D LIDAR points onto 2D camera images using KITTI calibration data.
- **3D Object Detection Visualization**: Parses KITTI label files to draw accurate 3D bounding boxes in the camera frame.
- **Multi-Object Tracking**: Tracks multiple objects over time using a Kalman filter-based tracking system.
- **Real-Time Animation**: Displays animated sensor fusion frames with bounding boxes and tracked object IDs.
- **Trajectory Visualization**: Maintains and visualizes the 2D trajectory of each tracked object across frames.
- **Pause and Play Controls**: User-friendly interactive animation controls.
- **Animation & Image Export**: Saves animation as GIF and trajectory summary images for further analysis.

---

## Project Structure

```


├── animate\_fusion.py                      # Main script for animation and visualization
├── kalman\_tracker.py                      # Multi-object Kalman filter tracking implementation
├── 2011\_09\_26\_drive\_0001\_sync/        # KITTI dataset files (images, LIDAR)
├── 2011\_09\_26/                           # KITTI calibration files
├── label\_2/                               # KITTI label files for object detection
├── README.md                               # This documentation file

````

---

## Getting Started

### Prerequisites

- Python 3.8+ (tested on Python 3.13)
- Packages:
  - numpy
  - matplotlib
  - opencv-python
  - filterpy

Install dependencies using pip:

```bash
pip install numpy matplotlib opencv-python filterpy
````

### Dataset Preparation

Download the relevant KITTI dataset sequences from [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/). Organize your folders as follows:

* Place synchronized LIDAR and image files in the `2011_09_26_drive_0001_sync` folder.
* Place calibration files in the `2011_09_26` folder.
* Place label files in the `label_2` folder.

Update the paths in `animate_fusion.py` accordingly if your folders are located elsewhere.

---

## How to Run

Run the main animation script:

```bash
python animate_fusion.py
```

A window will open displaying fused camera images with projected LIDAR points, 3D bounding boxes, and tracked object IDs. Use the **Pause** button to stop/start animation.

After closing the animation window, the program will automatically save:

* `fusion_tracking.gif` — animated sensor fusion and tracking visualization.
* `trajectory_summary.png` — snapshot image summarizing object trajectories.

---

## Key Components

### KalmanTracker & MultiObjectTracker

* Implements a Kalman filter-based tracking algorithm for 3D object tracking.
* Associates detections to existing tracks using nearest-neighbor distance thresholding.
* Creates new tracks for unmatched detections.
* Maintains object IDs and trajectories over time.

### Sensor Fusion and Visualization

* Projects 3D LIDAR points into the camera image frame using KITTI calibration.
* Parses object detection labels to draw accurate 3D bounding boxes.
* Uses OpenCV and Matplotlib for rendering images, points, and animations.

---

## Future Work

* **LLM Integration**: Add automated event summarization and anomaly reasoning using large language models.
* **Improved Data Association**: Use advanced algorithms like Hungarian or IoU-based matching.
* **Real-Time Performance Optimization**: Enhance efficiency for deployment on embedded systems.
* **Extended Object Types**: Support additional object classes and multi-sensor modalities.
* **User Interface Enhancements**: Add controls for frame navigation, object filtering, and detailed track inspection.

---

## License

This project is released under the MIT License.

---

## Acknowledgments

* KITTI Vision Benchmark Suite for dataset and calibration files.
* FilterPy for Kalman filter implementation.
* OpenCV and Matplotlib communities for visualization tools.

---

## Author

**Koutilya Ganapathiraju**  
Machine Learning Engineer | Robotics Enthusiast  
https://github.com/GKoutilya | www.linkedin.com/in/koutilya-ganapathiraju-0a3350182 | gkoutilyaraju@gmail.com  

Passionate about building cutting-edge perception and tracking systems for robotics and autonomous vehicles. Always eager to explore new AI techniques and share knowledge with the community.

For questions or suggestions, feel free to reach out.

---