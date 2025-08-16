import cv2
import numpy as np
from datetime import datetime
import os

# ===================== CONFIG =====================
cfg_path = "yolov4.cfg"
weights_path = "yolov4.weights"
video_path = "traffic.mp4"
conf_threshold = 0.5
nms_threshold = 0.4
vehicle_classes = ["bicycle", "car", "motorbike", "bus", "truck"]
MAX_DIST = 50

# ===================== LOAD YOLO =====================
print("[INFO] Loading YOLOv4...")
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# ===================== VIDEO SOURCE =====================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("[ERROR] Could not open video source")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# ===================== LANES =====================
LANE1_END_X = width // 3
LANE2_END_X = 2 * (width // 3)

def assign_lane(cx):
    if cx < LANE1_END_X: return 1
    elif cx < LANE2_END_X: return 2
    else: return 3

line_y = int(height * 0.6)
lane_lines = {1: line_y, 2: line_y+10, 3: line_y+20}
lane_counts = {1:0,2:0,3:0}
counted_vehicles = {1:set(),2:set(),3:set()}

# Lane colors for overlay
lane_colors = {1:(0,255,0),2:(0,255,255),3:(255,0,0)}

# ===================== CREATE OUTPUT FOLDER =====================
output_dir = "output_videos"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"vehicle_count_{timestamp}.avi")

fourcc = cv2.VideoWriter_fourcc(*"XVID")
video_out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
print(f"[INFO] Saving counted video to {output_file}")

# ===================== TRACKING =====================
vehicle_id_counter = 0
tracked_vehicles = {}

# ===================== PROCESS FRAMES =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    overlay = frame.copy()
    # Draw translucent lane backgrounds
    cv2.rectangle(overlay, (0,0), (LANE1_END_X,height), lane_colors[1], -1)
    cv2.rectangle(overlay, (LANE1_END_X,0), (LANE2_END_X,height), lane_colors[2], -1)
    cv2.rectangle(overlay, (LANE2_END_X,0), (width,height), lane_colors[3], -1)
    frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)

    # YOLO detection
    blob = cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True,crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [],[],[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and classes[class_id] in vehicle_classes:
                cx = int(detection[0]*width)
                cy = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(cx - w/2)
                y = int(cy - h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,conf_threshold,nms_threshold)
    current_centroids = []

    for i in indexes.flatten():
        x,y,w,h = boxes[i]
        cx = x + w//2
        cy = y + h//2
        lane = assign_lane(cx)
        current_centroids.append((cx,cy,x,y,w,h,lane))

    new_tracked = {}
    for cx,cy,x,y,w,h,lane in current_centroids:
        matched_id = None
        for vid,(tcx,tcy,tlane) in tracked_vehicles.items():
            if tlane==lane and abs(cx-tcx)<MAX_DIST and abs(cy-tcy)<MAX_DIST:
                matched_id=vid
                break
        if matched_id is None:
            vehicle_id_counter+=1
            matched_id = vehicle_id_counter

        prev_cy = tracked_vehicles.get(matched_id,(cx,cy-10,lane))[1]

        if matched_id not in counted_vehicles[lane]:
            if cy>prev_cy:
                if prev_cy<lane_lines[lane]<=cy:
                    lane_counts[lane]+=1
                    counted_vehicles[lane].add(matched_id)
            else:
                if prev_cy>lane_lines[lane]>=cy:
                    lane_counts[lane]+=1
                    counted_vehicles[lane].add(matched_id)

        new_tracked[matched_id]=(cx,cy,lane)

        # Draw vehicle rectangle and info
        color=(0,0,0)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,f"ID {matched_id}",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
        cv2.putText(frame,f"Lane {lane}",(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,0.6,lane_colors[lane],2)
        cv2.arrowedLine(frame,(cx,cy),(cx,cy-15),(0,0,255),2)

    tracked_vehicles = new_tracked

    # Draw lane dividers
    cv2.line(frame,(LANE1_END_X,0),(LANE1_END_X,height),(0,0,0),2)
    cv2.line(frame,(LANE2_END_X,0),(LANE2_END_X,height),(0,0,0),2)

    # Draw counting lines
    for l,y_line in lane_lines.items():
        cv2.line(frame,(0,y_line),(width,y_line),(0,0,255),2)

    # Display lane counts
    cv2.putText(frame,f"Lane1: {lane_counts[1]}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
    cv2.putText(frame,f"Lane2: {lane_counts[2]}",(LANE1_END_X+20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
    cv2.putText(frame,f"Lane3: {lane_counts[3]}",(LANE2_END_X+20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

    cv2.imshow("Traffic Flow - 3 Lanes Auto Save", frame)
    video_out.write(frame)

    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
video_out.release()
cv2.destroyAllWindows()
