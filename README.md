Lane Following System – Computer Vision Project (OpenCV)

This project implements a classical computer vision–based lane detection and following system using Python and OpenCV. It simulates the core logic used in autonomous vehicles for maintaining lane discipline, all using software-only methods (no physical robot required).

Project Overview

The system processes dashcam or road-driving videos to:

Detect lane boundaries using color and edge information.

Apply perspective transforms to obtain a top-down “bird’s-eye” view.

Fit polynomial curves to the left and right lane lines.

Estimate the vehicle’s lateral offset and steering direction.

Overlay an annotated driving zone (green) showing the current detected lane area.

Core Components

Preprocessing & Filtering

Converts each frame to multiple color spaces (HLS, LAB, GRAY).

Applies thresholding and Sobel edge detection to extract lane lines.

Suppresses noise like shadows, cracks, or road textures.

Region of Interest (ROI)

Focuses processing on the road section relevant to lane detection.

Ignores irrelevant areas (sky, dashboard, roadside).

Perspective Transform

Transforms the camera view into a top-down perspective for better geometric analysis.

Lane Line Detection

Uses sliding window and search-around-poly techniques to find lane pixels.

Handles both solid and dashed lines with adjustable tolerances.

Lane Smoothing & Validation

Applies exponential smoothing and geometric sanity checks.

Keeps the lane stable between frames, even if one side is momentarily lost.

Visualization

Draws a colored polygon (green) over the detected lane.

Displays real-time metrics such as offset from center and steering angle.

Enhancements Implemented

Added a lane regularization module to stabilize lane width and center.

Improved robustness against dark cracks, shadows, and dashed lanes.

Implemented adaptive fallback logic when lane lines are partially missing.

Added a default “safe zone” in case the lane is lost entirely.

Introduced color coding (green for safe, red for obstacle zones).

Limitations & Future Work

The model requires parameter tuning per video (e.g., brightness, contrast, lane width).

Performance may degrade in different lighting conditions, wet roads, or unfamiliar camera angles.

A future improvement would be to integrate deep learning (e.g., YOLOv5, U-Net) for adaptive lane detection that generalizes across diverse environments.

Technologies Used

Python

OpenCV (for image processing and computer vision)

NumPy (for matrix and numerical operations)
