import os
import cv2
import numpy as np
from pathlib import Path

def is_in_zone(x:int, y:int, zone:np.ndarray) -> bool:
    return cv2.pointPolygonTest(zone, (int(x), int(y)), measureDist=False) >= 0

def extract_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return np.array(frames)

def is_in_zone_vec(centers:np.ndarray, zone:np.ndarray) -> np.ndarray:
    return np.array([
        is_in_zone(x, y, zone)
        for x, y in centers
    ]).astype(bool)
    
def take_drive_pics(record, vid_path, drive_pics_folder_path, driving_region_pix):
    os.makedirs(drive_pics_folder_path, exist_ok=True)
    
    record = record.copy()
    record['in_driving_region'] = is_in_zone_vec(record[['cx', 'cy']].values.astype(int), driving_region_pix)
    frames = extract_frames(vid_path)
    for i, row in record.iterrows():
        if row['in_driving_region']:
            frame = frames[row['iteration']]
            if not np.isnan(row["tl_y"]) and not np.isnan(row["tl_x"]) and not np.isnan(row["h"]) and not np.isnan(row["w"]):
                cropped = frame[int(row["tl_y"]):(int(row["tl_y"]) + int(row["h"])),
                                int(row["tl_x"]):(int(row["tl_x"]) + int(row["w"]))]
                cv2.imwrite(drive_pics_folder_path + f"frame{i}.jpg", cropped)

def average_color_histogram(folder_path):
    folder_path = Path(folder_path)
    hist_sum = None
    count = 0

    for image_path in folder_path.glob("*.jpg"):
        image = cv2.imread(str(image_path))

        # Convert image to HSV for better color representation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for each channel
        hist_h = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
        hist_s = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

        # Concatenate histograms into one
        hist = np.concatenate((hist_h, hist_s, hist_v), axis=0)

        # Add to sum
        if hist_sum is None:
            hist_sum = hist
        else:
            hist_sum += hist
        count += 1

    if count == 0:
        return None
    
    return hist_sum / count