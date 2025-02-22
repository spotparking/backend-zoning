import numpy as np 
import pandas as pd
from pathlib import Path
import cv2
from car_class import Car
from zone_class import Zone
from helpers import take_drive_pics, is_in_zone_vec

def create_features(frames:list[np.ndarray], record:pd.DataFrame, zone_driving_region_coordinates:list[list[float]], license_plate:str) -> np.ndarray:
    
    ave_hist = average_color_histogram(record, frames, zone_driving_region_coordinates)
    center_pt = get_center_pt(record, zone_driving_region_coordinates)
    
    # Other future features (to be saved in the car class):
        # average width height at each driving picture
        # saved cropped images for classifier that takes 2 images and says if they have the same car in them
    
    return np.array([ave_hist, center_pt])

def average_color_histogram(record, frames, driving_region_coordinates):
    cropped_images = take_drive_pics(record, frames, driving_region_coordinates)
    
    hist_sum = None
    count = 0

    for image in cropped_images:
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
    
    avg_hist = hist_sum / count
    return avg_hist

def get_center_pt(record, driving_region_pix):
    record = record.copy()
    record['in_driving_region'] = is_in_zone_vec(record[['cx', 'cy']].values.astype(int), driving_region_pix)
    
    last_center_pt = None
    for i, row in record.iterrows():
        if row['in_driving_region']:
            last_center_pt = row[["cx", "cy"]]
    return last_center_pt