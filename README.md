# Hybrid-SOT-Trajectory-Predictor

A hybrid Single Object Tracking (SOT) and trajectory prediction framework designed for robust object tracking and future motion estimation in dynamic environments.

## Features

- Real-time single object tracking
- Future trajectory prediction
- Hybrid tracking-prediction architecture
- PyTorch implementation
- Evaluation and visualization tools

## Demo

<p align="center">
  <img src="assets/demo.gif" width="700">
</p>

## Method Overview

[Architecture Figure]

The proposed framework combines:

1. Object Detection
2. Single Object Tracking
3. Temporal Feature Extraction
4. Trajectory Prediction Network

## Project Structure

```text
Hybrid-SOT-Trajectory-Predictor/
│
├── datasets/
├── models/
├── tracker/
├── predictor/
├── utils/
├── train.py
├── test.py
└── README.md
