# cctv-human-detector
## CCTV Real-time Person Detection and AI Analysis with OpenCV and OpenVINO for Surveillance

### Description:

This Python project implements real-time person detection and analysis for surveillance applications using OpenCV and OpenVINO. It leverages the following functionalities:

### Motion Detection:

Captures video stream from a webcam or IP camera using OpenCV.
Compares consecutive frames to identify movement regions.

### Person Detection with AI Analysis (if motion detected):

Analize the frame with an Intel pre-trained model (FP16 instance-segmentation-person-0007).

### Process both sources:

Compares the bounding boxes from Motion Detection and IA Predictions to get an intersection that meets the requisites to trigger an action

This repository provides a foundation for building intelligent surveillance systems that can effectively detect people, reduce false positives, and run efficiently on resource-constrained devices, making it suitable for edge deployments.

Keywords: Python, OpenCV, OpenVINO, Person Detection, AI Analysis, Real-time, Surveillance, Webcam, IP Camera, False Alarm Reduction
