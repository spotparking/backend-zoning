import os

import cv2
import numpy as np

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