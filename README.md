# Traffic Flow Analysis — 3 Lane Vehicle Counting

## Overview
This project uses YOLOv8 + SORT tracking to detect and count vehicles in three distinct lanes of a traffic video.

## Features
- Vehicle detection with YOLOv8
- SORT tracking
- Lane counting (custom/equal lanes)
- Outputs annotated MP4, CSV log, summary PNG

## Setup
```
pip install ultralytics opencv-python pandas numpy yt-dlp sort-tracker
```

## Usage
Preset for target video:
```
python traffic_flow_counter.py --youtube https://www.youtube.com/watch?v=MNn9qKG2UFI
```
