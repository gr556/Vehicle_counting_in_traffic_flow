# Traffic Flow Analysis — 3 Lane Vehicle Counting

## Overview
This project uses YOLOv8 for real-time vehicle detection and SORT tracking for consistent IDs to count vehicles passing through three separate lanes in a traffic video.

It generates annotated videos, CSV logs, and summary snapshots for lane-wise traffic analysis.

## Features

* Vehicle Tracking: Unique IDs with SORT tracker.
* 3-Lane Counting: Accurate lane-wise vehicle counts.
* Outputs:
   * Annotated MP4 video
   * CSV log of lane counts
   * Summary PNG snapshot
* YouTube Video Support: Direct processing from YouTube URLs.
* Cross-platform: Works on Windows, macOS, and Linux.

## How it works

Video Input: Reads local or YouTube video.

Vehicle Detection: YOLOv8 detects vehicles frame-by-frame.

Tracking: SORT tracker assigns unique IDs to vehicles.

Lane Assignment: Vehicles assigned to lanes based on x-coordinate.

Counting: Vehicles counted when crossing lane line.

Visualization: Draws bounding boxes, lane lines, movement arrows, and counts.

Save Outputs: Annotated video, CSV log, and summary snapshot.

## Setup

Clone Repository
```
git clone https://github.com/your-username/traffic-flow-analysis-3lanes.git
cd traffic-flow-analysis-3lanes
```
Install Dependencies
```
pip install ultralytics opencv-python pandas numpy yt-dlp sort-tracker

```

## Usage
Preset for target video:
```
python traffic_flow_counter.py --youtube https://www.youtube.com/watch?v=MNn9qKG2UFI
```
Run on local video:
```
python traffic_flow_counter.py --video traffic.mp4 --display
```

## Future Improvements

Real-time dashboard for live traffic monitoring.

Automatic lane detection from camera perspective.

Integration with traffic signal analytics.
