import os
from pathlib import Path

import cv2
import numpy as np


###################
# FEATURE HELPERS #
###################
def is_in_zone(x:int, y:int, zone:np.ndarray) -> bool:
    return cv2.pointPolygonTest(zone, (int(x), int(y)), measureDist=False) >= 0

def is_in_zone_vec(centers:np.ndarray, zone:np.ndarray) -> np.ndarray:
    return np.array([
        is_in_zone(x, y, zone)
        for x, y in centers
    ]).astype(bool)
    
def take_drive_pics(record, frames, driving_region_pix):
    record = record.copy()
    record['in_driving_region'] = is_in_zone_vec(record[['cx', 'cy']].values.astype(int), driving_region_pix)
    
    cropped_images = []
    for i, row in record.iterrows():
        if row['in_driving_region']:
            frame = frames[row['iteration']]
            if not np.isnan(row["tl_y"]) and not np.isnan(row["tl_x"]) and not np.isnan(row["h"]) and not np.isnan(row["w"]):
                cropped = frame[int(row["tl_y"]):(int(row["tl_y"]) + int(row["h"])),
                                int(row["tl_x"]):(int(row["tl_x"]) + int(row["w"]))]
                cropped_images.append(cropped)
    return cropped_images

def get_images_from_folder(folder_path):
    folder_path = Path(folder_path)
    images = [
        cv2.imread(str(image_path)) 
        for image_path in folder_path.glob("*.jpg")]
    return images

def points_float_to_pix(points:list[list[float]], height:int, width:int) -> list[list[int]]:
    """ takes in a list of points like [[0.1, 0.2], [0.3, 0.4]] and the height and width
    of the image, and returns a list of points like [[100, 200], [300, 400]] """
    return [[int(point[0] * width), int(point[1] * height)] for point in points]


######################
# PREDICTION HELPERS #
######################
def closest_k_points(leave, enter_points, k):
    """
    Used to find the closest k entering instances compared to the leaving instance
    """
    # Compute distances and store them with points
    distances = [(point, np.linalg.norm(np.array(point) - np.array(leave))) for point in enter_points]
    
    # Sort by distance and return the k closest points
    return [point for point, _ in sorted(distances, key=lambda x: x[1])[:k]]

def compare_histograms(hist1, hist2):
    hist1 = hist1.astype(np.float32)
    hist2 = hist2.astype(np.float32)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
